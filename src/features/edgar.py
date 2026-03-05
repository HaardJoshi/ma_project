"""
SEC EDGAR 10-K Filing Pipeline.

4-stage pipeline to fetch and extract text from SEC EDGAR filings:
  Stage 1: Bloomberg ticker → SEC CIK mapping
  Stage 2: Filing discovery (find 10-K before announcement date)
  Stage 3: Document download (full 10-K HTML)
  Stage 4: Section extraction (Item 7 MD&A, Item 1A Risk Factors)

SEC EDGAR rate limit: 10 requests/second.
Requires User-Agent header with name + email (SEC policy).
"""

import csv
import gzip
import json
import os
import re
import time
import logging
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)

# ─── SEC API endpoints ──────────────────────────────────────────────────────────
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"
SEC_FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
SEC_COMPANY_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{query}%22&dateRange=custom&startdt=2000-01-01&enddt=2025-12-31&forms=10-K"

# ─── Rate limiter ───────────────────────────────────────────────────────────────
_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 0.12  # ~8 req/sec to stay safely under 10/sec limit


def _rate_limited_get(url: str, user_agent: str, max_retries: int = 3) -> bytes:
    """
    Make a rate-limited HTTP GET request to SEC EDGAR.

    Implements exponential backoff on HTTP 429 / 5xx errors.
    """
    global _last_request_time

    for attempt in range(max_retries):
        # Rate limiting
        elapsed = time.time() - _last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        req = Request(url)
        req.add_header("User-Agent", user_agent)
        req.add_header("Accept-Encoding", "gzip, deflate")

        try:
            _last_request_time = time.time()
            with urlopen(req, timeout=30) as resp:
                raw = resp.read()
                # Decompress gzip if needed
                encoding = resp.headers.get("Content-Encoding", "")
                if encoding == "gzip" or (raw[:2] == b"\x1f\x8b"):
                    raw = gzip.decompress(raw)
                return raw
        except HTTPError as e:
            if e.code == 429 or e.code >= 500:
                wait = 2 ** (attempt + 1)
                logger.warning(f"HTTP {e.code} on {url}, retrying in {wait}s...")
                time.sleep(wait)
            elif e.code == 404:
                logger.debug(f"HTTP 404: {url}")
                return b""
            else:
                logger.error(f"HTTP {e.code} on {url}: {e.reason}")
                raise
        except (URLError, TimeoutError) as e:
            wait = 2 ** (attempt + 1)
            logger.warning(f"Network error on {url}: {e}, retrying in {wait}s...")
            time.sleep(wait)

    logger.error(f"Failed after {max_retries} retries: {url}")
    return b""


# ═════════════════════════════════════════════════════════════════════════════════
# Stage 1: Ticker → CIK Mapping
# ═════════════════════════════════════════════════════════════════════════════════

def _bloomberg_to_standard(ticker: str) -> list[str]:
    """
    Convert Bloomberg ticker (e.g. 'AAPL US') to standard ticker variants.

    Returns multiple candidates to improve match rate:
    - Base ticker ('AAPL')
    - Without trailing Q/D (delisted markers)
    """
    base = ticker.strip()
    if base.endswith(" US"):
        base = base[:-3].strip()

    candidates = [base]

    # Delisted: try without trailing Q or D
    if len(base) > 1 and base[-1] in ("Q", "D") and base[:-1].isalpha():
        candidates.append(base[:-1])

    return candidates


def build_ticker_cik_map(
    user_agent: str,
    cache_path: Optional[Path] = None,
) -> dict[str, int]:
    """
    Download SEC company_tickers.json and build ticker → CIK mapping.

    Parameters
    ----------
    user_agent : str
        Required by SEC (e.g. "Jane Doe jane@uni.ac.uk").
    cache_path : Path, optional
        Where to cache. Defaults to data/external/edgar/ticker_cik_map.json.

    Returns
    -------
    dict[str, int]
        Mapping of uppercase ticker → CIK number.
    """
    if cache_path is None:
        cache_path = PROJECT_ROOT / "data" / "external" / "edgar" / "ticker_cik_map.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Use cache if fresh (< 24h old)
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:
            logger.info(f"Using cached ticker→CIK map ({age_hours:.1f}h old)")
            with open(cache_path) as f:
                return json.load(f)

    logger.info("Downloading SEC company_tickers.json...")
    data = _rate_limited_get(SEC_TICKERS_URL, user_agent)
    if not data:
        raise RuntimeError("Failed to download company_tickers.json from SEC")

    raw = json.loads(data)
    # raw format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc"}, ...}
    mapping = {}
    for entry in raw.values():
        ticker = entry.get("ticker", "").upper().strip()
        cik = entry.get("cik_str")
        if ticker and cik:
            mapping[ticker] = int(cik)

    with open(cache_path, "w") as f:
        json.dump(mapping, f, indent=2)

    logger.info(f"Ticker→CIK map: {len(mapping)} entries, cached → {cache_path}")
    return mapping


def resolve_tickers(
    acquirer_tickers: list[str],
    ticker_cik_map: dict[str, int],
    output_dir: Optional[Path] = None,
) -> tuple[dict[str, int], list[str]]:
    """
    Resolve Bloomberg tickers to CIK numbers.

    Returns
    -------
    tuple[dict[str, int], list[str]]
        (matched: {bloomberg_ticker → CIK}, unmatched: [tickers])
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "external" / "edgar"
    output_dir.mkdir(parents=True, exist_ok=True)

    matched = {}
    unmatched = []

    for bt in acquirer_tickers:
        bt = bt.strip()
        if not bt or not bt.endswith(" US"):
            continue  # skip non-US tickers

        # Skip Bloomberg numeric identifiers (e.g. '0167866D US')
        base = bt[:-3].strip()  # remove ' US'
        if base and base[0].isdigit():
            continue  # these are Bloomberg-internal IDs, not exchange tickers

        candidates = _bloomberg_to_standard(bt)
        found = False
        for candidate in candidates:
            candidate_upper = candidate.upper()
            if candidate_upper in ticker_cik_map:
                matched[bt] = ticker_cik_map[candidate_upper]
                found = True
                break
        if not found:
            unmatched.append(bt)

    # Save unmatched for review
    unmatched_path = output_dir / "unmatched_tickers.csv"
    with open(unmatched_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bloomberg_ticker", "standard_ticker_tried"])
        for bt in unmatched:
            candidates = _bloomberg_to_standard(bt)
            writer.writerow([bt, "|".join(candidates)])

    logger.info(f"Ticker resolution: {len(matched)} matched, {len(unmatched)} unmatched")
    logger.info(f"  (Bloomberg numeric IDs skipped — not real exchange tickers)")
    return matched, unmatched


# ═════════════════════════════════════════════════════════════════════════════════
# Stage 2: Filing Discovery
# ═════════════════════════════════════════════════════════════════════════════════

def _search_filings_array(
    forms: list, dates: list, accessions: list, primary_docs: list,
    target_date: str,
) -> Optional[dict]:
    """
    Search a filings array for the best 10-K before target_date.

    Prefers 10-K over 10-K/A. Among same form types, picks the most recent.
    """
    best_10k = None      # prefer regular 10-K
    best_10ka = None     # fallback to 10-K/A

    for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
        if form not in ("10-K", "10-K/A"):
            continue
        if date >= target_date:
            continue

        entry = {
            "accession": acc,
            "filing_date": date,
            "primary_doc": doc,
            "form_type": form,
        }

        if form == "10-K":
            if best_10k is None or date > best_10k["filing_date"]:
                best_10k = entry
        else:  # 10-K/A
            if best_10ka is None or date > best_10ka["filing_date"]:
                best_10ka = entry

    return best_10k or best_10ka  # prefer 10-K, fall back to 10-K/A


def find_10k_before_date(
    cik: int,
    announce_date: str,
    user_agent: str,
) -> Optional[dict]:
    """
    Find the most recent 10-K filing for a CIK before the given date.

    Prefers regular 10-K over 10-K/A (amended filings are often partial).
    Checks both the 'recent' filings array and older paginated filings.

    Parameters
    ----------
    cik : int
        SEC Central Index Key.
    announce_date : str
        ISO date string (YYYY-MM-DD).
    user_agent : str
        SEC User-Agent.

    Returns
    -------
    dict or None
        {"accession": str, "filing_date": str, "primary_doc": str} or None.
    """
    cik_padded = str(cik).zfill(10)
    url = SEC_SUBMISSIONS_URL.format(cik=cik_padded)
    data = _rate_limited_get(url, user_agent)
    if not data:
        return None

    try:
        submissions = json.loads(data)
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON from submissions for CIK {cik}")
        return None

    # ── Search the 'recent' filings array first ──────────────────────
    recent = submissions.get("filings", {}).get("recent", {})
    result = _search_filings_array(
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
        announce_date,
    )

    if result:
        return result

    # ── Check older paginated filings ────────────────────────────────
    # The SEC submissions API paginates older filings into separate JSON files
    # listed in filings.files[].name
    older_files = submissions.get("filings", {}).get("files", [])
    for file_entry in older_files:
        fname = file_entry.get("name", "")
        if not fname:
            continue

        older_url = f"https://data.sec.gov/submissions/{fname}"
        older_data = _rate_limited_get(older_url, user_agent)
        if not older_data:
            continue

        try:
            older = json.loads(older_data)
        except json.JSONDecodeError:
            continue

        result = _search_filings_array(
            older.get("form", []),
            older.get("filingDate", []),
            older.get("accessionNumber", []),
            older.get("primaryDocument", []),
            announce_date,
        )

        if result:
            return result

    return None


# ═════════════════════════════════════════════════════════════════════════════════
# Stage 3: Document Download
# ═════════════════════════════════════════════════════════════════════════════════

def _find_10k_document_url(
    cik: int,
    accession: str,
    primary_doc: str,
    user_agent: str,
) -> Optional[str]:
    """
    Find the actual 10-K document URL by examining the filing index.

    The primaryDocument from the submissions API often points to a small
    cover page. The actual 10-K content is usually the largest .htm file
    in the filing package.
    """
    accession_clean = accession.replace("-", "")
    index_url = SEC_FILING_INDEX_URL.format(cik=cik, accession=accession_clean)

    data = _rate_limited_get(index_url, user_agent)
    if not data:
        # Fallback to primaryDocument
        return SEC_ARCHIVES_URL.format(
            cik=cik, accession=accession_clean, filename=primary_doc
        )

    html = data.decode("utf-8", errors="replace")

    # Parse the index page 一 find all .htm/.html links and their sizes
    # The index page lists files in a table with columns: Name, Size, etc.
    # We want the largest .htm/.html file (excluding *-index.htm)
    htm_files = re.findall(
        r'href="([^"]+\.htm[l]?)"',
        html,
        re.IGNORECASE,
    )

    # Filter out index files and R-files (XBRL)
    candidates = [
        f for f in htm_files
        if not f.endswith("-index.htm")
        and not f.endswith("-index.html")
        and "/R" not in f
        and not f.startswith("R")
    ]

    if not candidates:
        # Fallback to primaryDocument
        return SEC_ARCHIVES_URL.format(
            cik=cik, accession=accession_clean, filename=primary_doc
        )

    # Try to find the actual 10-K document by checking file sizes
    # Download headers for each candidate to find the largest one
    best_url = None
    best_size = 0

    for fname in candidates:
        # Build full URL
        if fname.startswith("http"):
            url = fname
        else:
            url = SEC_ARCHIVES_URL.format(
                cik=cik, accession=accession_clean, filename=fname
            )

        # Quick HEAD-like check: download just the first bytes to test,
        # or we can just pick the primary doc if it looks right
        # For efficiency, just get the file and check size
        file_data = _rate_limited_get(url, user_agent)
        if file_data and len(file_data) > best_size:
            best_size = len(file_data)
            best_url = url
            best_data = file_data

    # Return the URL of the largest file
    return best_url if best_url else SEC_ARCHIVES_URL.format(
        cik=cik, accession=accession_clean, filename=primary_doc
    )


def download_10k(
    cik: int,
    filing_info: dict,
    user_agent: str,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Download the full 10-K document.

    Fetches the filing index page and selects the largest .htm file
    (which is the actual 10-K content, not a cover page).

    Returns
    -------
    Path or None
        Path to the saved HTML file, or None on failure.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "external" / "edgar" / "filings"

    accession_clean = filing_info["accession"].replace("-", "")
    filing_dir = output_dir / str(cik) / accession_clean
    filing_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta_path = filing_dir / "filing_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"cik": cik, **filing_info}, f, indent=2)

    # Check if already downloaded (skip if file > 50KB — likely the real doc)
    doc_path = filing_dir / "full_10k.html"
    if doc_path.exists() and doc_path.stat().st_size > 50_000:
        logger.debug(f"Already downloaded: {doc_path}")
        return doc_path

    # Strategy: Try primaryDocument first. If it's small (< 50KB),
    # it's likely just a cover page — fetch the filing index to find
    # the real document.
    primary_url = SEC_ARCHIVES_URL.format(
        cik=cik,
        accession=accession_clean,
        filename=filing_info["primary_doc"],
    )
    data = _rate_limited_get(primary_url, user_agent)

    if data and len(data) > 50_000:
        # Primary doc looks like the real thing
        with open(doc_path, "wb") as f:
            f.write(data)
        logger.debug(f"Downloaded {len(data):,} bytes (primary) → {doc_path}")
        return doc_path

    # Primary doc is too small — scan the filing index for the largest .htm
    logger.debug(f"Primary doc only {len(data):,} bytes, scanning filing index...")
    index_url = SEC_FILING_INDEX_URL.format(cik=cik, accession=accession_clean)
    index_data = _rate_limited_get(index_url, user_agent)

    if not index_data:
        # Can't get index; save what we have
        if data:
            with open(doc_path, "wb") as f:
                f.write(data)
            return doc_path
        return None

    index_html = index_data.decode("utf-8", errors="replace")

    # Find all .htm/.html filenames linked from the index
    htm_files = re.findall(
        r'href="([^"]+\.htm[l]?)"',
        index_html,
        re.IGNORECASE,
    )
    # Filter out index/R-files
    candidates = [
        f for f in htm_files
        if "index" not in f.lower()
        and not f.startswith("R")
        and not f.startswith("/")
    ]

    if not candidates:
        if data:
            with open(doc_path, "wb") as f:
                f.write(data)
            return doc_path
        return None

    # Download each candidate and keep the largest
    best_data = data  # start with what we already have
    for fname in candidates:
        url = SEC_ARCHIVES_URL.format(
            cik=cik, accession=accession_clean, filename=fname
        )
        candidate_data = _rate_limited_get(url, user_agent)
        if candidate_data and len(candidate_data) > len(best_data or b""):
            best_data = candidate_data
            logger.debug(f"  Found larger doc: {fname} ({len(candidate_data):,} bytes)")

    if best_data:
        with open(doc_path, "wb") as f:
            f.write(best_data)
        logger.debug(f"Downloaded {len(best_data):,} bytes → {doc_path}")
        return doc_path

    return None


# ═════════════════════════════════════════════════════════════════════════════════
# Stage 4: Section Extraction
# ═════════════════════════════════════════════════════════════════════════════════

class _HTMLTextExtractor(HTMLParser):
    """Simple HTML → plain text converter."""

    def __init__(self):
        super().__init__()
        self._text = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4"):
            self._text.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._text.append(data)

    def get_text(self) -> str:
        return "".join(self._text)


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
    except Exception:
        # Fallback: strip tags with regex
        return re.sub(r"<[^>]+>", " ", html)
    return parser.get_text()


# Regex patterns for section boundaries
# These look for "Item 7" or "Item 1A" headings in various formats
_ITEM_PATTERNS = {
    "item_7": [
        # "Item 7" with full title
        re.compile(
            r"item[\s\xa0]*7\.?\s*[-–—.\s]*management.{0,10}s?\s*discussion",
            re.IGNORECASE,
        ),
        # Just "Item 7" (with period or not)
        re.compile(r"item[\s\xa0]+7\.?\b", re.IGNORECASE),
        # Standalone heading: "MANAGEMENT'S DISCUSSION AND ANALYSIS"
        re.compile(
            r"management.{0,5}s\s+discussion\s+and\s+analysis",
            re.IGNORECASE,
        ),
    ],
    "item_1a": [
        # "Item 1A" with full title
        re.compile(
            r"item[\s\xa0]*1a\.?\s*[-–—.\s]*risk\s+factors",
            re.IGNORECASE,
        ),
        # Just "Item 1A"
        re.compile(r"item[\s\xa0]+1a\.?\b", re.IGNORECASE),
        # Standalone heading
        re.compile(r"risk\s+factors", re.IGNORECASE),
    ],
}

# Patterns marking the END of a section (start of next item)
_NEXT_ITEM = {
    "item_7": re.compile(
        r"item[\s\xa0]*7a\.?\s*[-–—.\s]*quantitative"
        r"|item[\s\xa0]*8\.?\s*[-–—.\s]*financial\s+statements",
        re.IGNORECASE,
    ),
    "item_1a": re.compile(
        r"item[\s\xa0]*1b\.?\s*[-–—.\s]*unresolved"
        r"|item[\s\xa0]*2\.?\s*[-–—.\s]*properties",
        re.IGNORECASE,
    ),
}

# Minimum content length for a valid section (chars)
MIN_SECTION_LEN = 200


def _extract_bounded(text: str, start: int, next_pattern) -> str:
    """Extract text from *start* to the next section marker (or max 80,000 chars)."""
    if next_pattern:
        # Search for end marker, starting at least 100 chars after start
        end_match = next_pattern.search(text, start + 100)
        if end_match:
            return text[start : end_match.start()]
    # No end marker — take up to 80,000 chars
    return text[start : start + 80_000]


def extract_section(text: str, section: str) -> Optional[str]:
    """
    Extract a section (item_7 or item_1a) from 10-K plain text.

    Uses a **longest-content-wins** strategy: finds ALL matches for the
    section header across all regex patterns, computes the bounded content
    for each, and returns the longest extraction. This avoids picking
    table-of-contents entries or cross-references.

    Parameters
    ----------
    text : str
        Full 10-K plain text.
    section : str
        'item_7' or 'item_1a'.

    Returns
    -------
    str or None
        Extracted section text, or None if not found.
    """
    patterns = _ITEM_PATTERNS.get(section, [])
    next_pattern = _NEXT_ITEM.get(section)

    # Collect ALL match positions from ALL patterns
    all_starts = []
    for pat in patterns:
        for m in pat.finditer(text):
            all_starts.append(m.start())

    if not all_starts:
        return None

    # De-duplicate starts that are very close together (within 50 chars)
    all_starts.sort()
    deduped = [all_starts[0]]
    for s in all_starts[1:]:
        if s - deduped[-1] > 50:
            deduped.append(s)

    # Try each start position, keep the one that produces the longest content
    best_text = None
    best_len = 0

    for start in deduped:
        candidate = _extract_bounded(text, start, next_pattern)
        candidate = re.sub(r"\n{3,}", "\n\n", candidate).strip()

        if len(candidate) > best_len:
            best_len = len(candidate)
            best_text = candidate

    if best_text is None or best_len < MIN_SECTION_LEN:
        logger.debug(
            f"Section {section}: best candidate only {best_len} chars "
            f"(tried {len(deduped)} positions), skipping"
        )
        return None

    logger.debug(
        f"Section {section}: extracted {best_len:,} chars "
        f"(best of {len(deduped)} candidates)"
    )
    return best_text


def extract_sections_from_file(html_path: Path) -> dict[str, Optional[str]]:
    """
    Extract Item 7 (MD&A) and Item 1A (Risk Factors) from a 10-K HTML file.

    Returns
    -------
    dict
        {"item_7_mda": str|None, "item_1a_risk": str|None}
    """
    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()

    text = _html_to_text(html)

    return {
        "item_7_mda": extract_section(text, "item_7"),
        "item_1a_risk": extract_section(text, "item_1a"),
    }


# ═════════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════════════

def _load_download_log(log_path: Path) -> set[str]:
    """Load the set of already-processed deal keys from the checkpoint log."""
    done = set()
    if log_path.exists():
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row.get("deal_key", ""))
    return done


def _append_download_log(log_path: Path, row: dict) -> None:
    """Append one row to the download checkpoint log."""
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_pipeline(
    user_agent: str,
    cleaned_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
    limit: Optional[int] = None,
    resume: bool = True,
) -> dict:
    """
    Run the full EDGAR pipeline.

    Parameters
    ----------
    user_agent : str
        SEC-required header (e.g. "Jane Doe jane@uni.ac.uk").
    cleaned_csv : str, optional
        Path to ma_cleaned.csv. Defaults to data/interim/ma_cleaned.csv.
    output_dir : str, optional
        Root output dir. Defaults to data/external/edgar.
    limit : int, optional
        Max number of deals to process (for testing).
    resume : bool
        If True, skip already-processed deals.

    Returns
    -------
    dict
        Summary statistics.
    """
    cleaned_csv = Path(cleaned_csv) if cleaned_csv else PROJECT_ROOT / "data" / "interim" / "ma_cleaned.csv"
    output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "data" / "external" / "edgar"
    filings_dir = output_dir / "filings"
    output_dir.mkdir(parents=True, exist_ok=True)
    filings_dir.mkdir(parents=True, exist_ok=True)

    download_log_path = output_dir / "download_log.csv"
    extraction_log_path = output_dir / "extraction_log.csv"

    # ── Stage 1: Build ticker → CIK mapping ─────────────────────────
    print("=" * 60)
    print("Stage 1: Ticker → CIK Mapping")
    print("=" * 60)

    ticker_cik_map = build_ticker_cik_map(user_agent, output_dir / "ticker_cik_map.json")
    print(f"  SEC ticker database: {len(ticker_cik_map)} entries")

    # Read deals from cleaned CSV
    with open(cleaned_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        deals = list(reader)

    # Filter to US acquirers only
    us_deals = [d for d in deals if d.get("Acquirer Ticker", "").strip().endswith(" US")]
    print(f"  Total deals: {len(deals)}, US acquirer deals: {len(us_deals)}")

    # Resolve tickers
    unique_tickers = list(set(d["Acquirer Ticker"].strip() for d in us_deals))
    matched, unmatched = resolve_tickers(unique_tickers, ticker_cik_map, output_dir)
    print(f"  Matched: {len(matched)}, Unmatched: {len(unmatched)}")

    # ── Load checkpoint ──────────────────────────────────────────────
    done_keys = _load_download_log(download_log_path) if resume else set()
    if done_keys:
        print(f"  Resuming: {len(done_keys)} deals already processed")

    # ── Stages 2–4: Process each deal ────────────────────────────────
    print("\n" + "=" * 60)
    print("Stages 2–4: Discovery, Download, Extraction")
    print("=" * 60)

    stats = {
        "total_us_deals": len(us_deals),
        "cik_matched": 0,
        "filing_found": 0,
        "downloaded": 0,
        "item_7_extracted": 0,
        "item_1a_extracted": 0,
        "skipped_resume": 0,
        "errors": 0,
    }

    deals_to_process = us_deals[:limit] if limit else us_deals

    for i, deal in enumerate(deals_to_process):
        ticker = deal.get("Acquirer Ticker", "").strip()
        announce_date = deal.get("Announce Date", "").strip()
        deal_key = f"{ticker}|{announce_date}"

        if deal_key in done_keys:
            stats["skipped_resume"] += 1
            continue

        # Progress
        if (i + 1) % 50 == 0 or i == 0:
            print(f"\n  Processing deal {i+1}/{len(deals_to_process)}: "
                  f"{ticker} (announced {announce_date})")

        # Check CIK
        cik = matched.get(ticker)
        if cik is None:
            _append_download_log(download_log_path, {
                "deal_key": deal_key, "ticker": ticker, "announce_date": announce_date,
                "cik": "", "status": "no_cik", "accession": "", "filing_date": "",
                "item_7": False, "item_1a": False,
            })
            continue
        stats["cik_matched"] += 1

        # Stage 2: Find 10-K
        try:
            filing = find_10k_before_date(cik, announce_date, user_agent)
        except Exception as e:
            logger.error(f"Error finding filing for {ticker} (CIK {cik}): {e}")
            stats["errors"] += 1
            _append_download_log(download_log_path, {
                "deal_key": deal_key, "ticker": ticker, "announce_date": announce_date,
                "cik": cik, "status": "discovery_error", "accession": "", "filing_date": "",
                "item_7": False, "item_1a": False,
            })
            continue

        if filing is None:
            _append_download_log(download_log_path, {
                "deal_key": deal_key, "ticker": ticker, "announce_date": announce_date,
                "cik": cik, "status": "no_10k_found", "accession": "", "filing_date": "",
                "item_7": False, "item_1a": False,
            })
            continue
        stats["filing_found"] += 1

        # Stage 3: Download
        try:
            html_path = download_10k(cik, filing, user_agent, filings_dir)
        except Exception as e:
            logger.error(f"Download error for {ticker}: {e}")
            stats["errors"] += 1
            _append_download_log(download_log_path, {
                "deal_key": deal_key, "ticker": ticker, "announce_date": announce_date,
                "cik": cik, "status": "download_error",
                "accession": filing["accession"], "filing_date": filing["filing_date"],
                "item_7": False, "item_1a": False,
            })
            continue

        if html_path is None:
            _append_download_log(download_log_path, {
                "deal_key": deal_key, "ticker": ticker, "announce_date": announce_date,
                "cik": cik, "status": "download_failed",
                "accession": filing["accession"], "filing_date": filing["filing_date"],
                "item_7": False, "item_1a": False,
            })
            continue
        stats["downloaded"] += 1

        # Stage 4: Extract sections
        try:
            sections = extract_sections_from_file(html_path)
        except Exception as e:
            logger.error(f"Extraction error for {ticker}: {e}")
            stats["errors"] += 1
            sections = {"item_7_mda": None, "item_1a_risk": None}

        filing_dir = html_path.parent
        has_item7 = False
        has_item1a = False

        if sections["item_7_mda"]:
            with open(filing_dir / "item_7_mda.txt", "w", encoding="utf-8") as f:
                f.write(sections["item_7_mda"])
            stats["item_7_extracted"] += 1
            has_item7 = True

        if sections["item_1a_risk"]:
            with open(filing_dir / "item_1a_risk.txt", "w", encoding="utf-8") as f:
                f.write(sections["item_1a_risk"])
            stats["item_1a_extracted"] += 1
            has_item1a = True

        _append_download_log(download_log_path, {
            "deal_key": deal_key, "ticker": ticker, "announce_date": announce_date,
            "cik": cik, "status": "success",
            "accession": filing["accession"], "filing_date": filing["filing_date"],
            "item_7": has_item7, "item_1a": has_item1a,
        })

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<25s}: {v}")
    print("=" * 60)

    return stats
