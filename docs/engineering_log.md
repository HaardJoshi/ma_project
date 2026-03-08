# SEC EDGAR Pipeline — Engineering Log

Running log of issues, root causes, and fixes encountered during development.

---

## Issue #1: `UnicodeDecodeError` on SEC JSON download
**Date:** 2026-03-05  
**Stage:** 1 (Ticker → CIK Mapping)  
**Symptom:** `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 1`  
**Root cause:** The request included `Accept-Encoding: gzip` but the response wasn't being decompressed. Byte `0x1f 0x8b` is the gzip magic number.  
**Fix:** Added `gzip.decompress()` in `_rate_limited_get()` when `Content-Encoding: gzip` header is present or the response starts with the gzip magic bytes.

---

## Issue #2: Low ticker match rate (998/2,467 = 40%)
**Date:** 2026-03-05  
**Stage:** 1 (Ticker → CIK Mapping)  
**Symptom:** 1,469 tickers reported as "unmatched".  
**Root cause:** Two distinct sub-populations:
1. **Bloomberg numeric IDs** (e.g. `0167866D US`, `9993232D US`) — these are internal Bloomberg identifiers for private/delisted companies, not real exchange tickers. ~420 of these.
2. **Genuinely delisted tickers** (e.g. `AET US`, `TWX US`) — companies acquired or renamed before the SEC's current `company_tickers.json` snapshot. These *had* valid CIKs historically but aren't in the current ticker list.  

**Fix (partial):** Skipped Bloomberg numeric IDs (base ticker starts with a digit). Reduced unmatched to ~1,049. The remaining ~1,049 are genuinely delisted tickers — a full fix would require a historical ticker→CIK database (future improvement).

---

## Issue #3: Only 1/5 filings found (`no_10k_found` for 3 deals)
**Date:** 2026-03-05  
**Stage:** 2 (Filing Discovery)  
**Symptom:** `T US` (announced 2016), `BMY US` (2019), `PFE US` (1999) returned `no_10k_found`.  
**Root cause:** The SEC submissions API paginates filings — the `recent` array only holds the last ~1,000 filings. For large filers, 10-K filings from before 2018 are in paginated JSON files listed under `filings.files[].name`.  
**Fix:** Added pagination support — after checking `recent`, the code now iterates through `filings.files[]`, downloads each paginated JSON, and searches for 10-K filings there.

---

## Issue #4: Downloaded 10-K/A instead of 10-K
**Date:** 2026-03-05  
**Stage:** 2 (Filing Discovery)  
**Symptom:** RTX US (CIK 101829) matched to a `10-K/A` (amended annual report) which was only a partial filing amendment.  
**Root cause:** `find_10k_before_date` treated `10-K` and `10-K/A` equally, picking whichever was most recent. `10-K/A` filings often contain only the amended sections, not the full annual report.  
**Fix:** Added two-tier preference — `_search_filings_array()` now tracks `best_10k` and `best_10ka` separately, returning the regular 10-K if available, falling back to 10-K/A only if no regular 10-K exists.

---

## Issue #5: Downloaded file was cover page (9KB text), not full 10-K
**Date:** 2026-03-05  
**Stage:** 3 (Document Download)  
**Symptom:** `full_10k.html` was 67KB of HTML but only 9,218 chars of text — a form cover page, not the actual 10-K.  
**Root cause:** The `primaryDocument` field from the SEC submissions API sometimes points to the filing wrapper/cover page rather than the actual 10-K document body.  
**Fix:** `download_10k()` now checks if the primary document is < 50KB. If so, it fetches the filing index page, finds all `.htm`/`.html` files, downloads each candidate, and keeps the largest one (which is the actual 10-K content). This correctly found 4.4MB and 568KB documents in testing.

---

## Issue #6: Item 7 (MD&A) extraction too short (264–358 chars)
**Date:** 2026-03-05  
**Stage:** 4 (Section Extraction)  
**Symptom:** 2 of 4 downloaded filings had Item 7 flagged as "too short" (< 500 chars). Only 1/4 Item 7 extracted vs 3/4 Item 1A.  
**Root cause:** The regex uses the LAST match of "Item 7" in the document, assuming table-of-contents entries come first. But some filings have a summary/cross-reference near the end that also mentions "Item 7", producing a short match between the summary entry and the next section marker.  
**Fix (in progress):** Switching to a **longest-content-wins** strategy — find ALL matches of the section header, compute the content length for each, and return the one that produces the longest extraction.

---

## Entry #7: FinBERT pipeline completed successfully
**Date:** 2026-03-08  
**Stage:** Block B (Text Embeddings)  
**Result:**
- 1,678 deals processed on Apple Silicon MPS device
- MD&A: 1,674 valid embeddings, PCA (64 components) explains 96.5% variance
- Risk Factors: 1,482 valid embeddings, PCA (64 components) explains 96.8% variance  
- Output: `data/processed/text_embeddings.csv` (128 dims per deal)
- Raw embeddings backed up to `data/processed/raw_embeddings.npz`  
**Notes:** The "UNEXPECTED keys" warning during FinBERT loading is harmless — we load `AutoModel` (transformer only) from a model saved with a classification head. Token length warning (>512) is expected and handled by chunking.

---

## Known Limitations

| Issue | Impact | Potential Fix |
|---|---|---|
| ~1,049 delisted tickers unmapped | ~20% of US deals missing | Historical ticker→CIK database |
| Section extraction heuristic | ~80-90% expected success rate | ML-based section classifier |
| Non-US acquirers skipped | ~3% of deals | Out of scope (no EDGAR filings) |
