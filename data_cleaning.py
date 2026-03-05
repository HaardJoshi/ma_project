pa"""
M&A Data Cleaning Pipeline
===========================
Reads  : ma_combined.csv   (4,999 rows × 77 cols)
Writes : ma_cleaned.csv          – cleaned tabular features
         ma_deal_descriptions.csv – deal descriptions for NLP (Block B)

Uses only the Python standard library (no pandas).
"""

import csv
import os
import re
from datetime import datetime

# ─── paths ──────────────────────────────────────────────────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__))
INPUT   = os.path.join(DIR, "ma_combined.csv")
OUTPUT  = os.path.join(DIR, "ma_cleaned.csv")
NLP_OUT = os.path.join(DIR, "ma_deal_descriptions.csv")

# ─── Step 1: columns to DROP (constant / useless) ──────────────────────────────
DROP_CONSTANT = {
    "Deal Type",          # always "M&A"
    "Deal Status",        # always "Completed"
    "Deal Description",   # → separate NLP file
    "Legal Adviser to Target Financial",  # categorical, not a financial feature
}

# ─── Step 2: columns to DROP (fill rate < 5%) ──────────────────────────────────
DROP_SPARSE = {
    "Seller Name",                                        # 2.0 %
    "Seller Ticker",                                      # 1.6 %
    "Target Current Market Cap",                          # 3.4 %
    "Target Price Earnings Ratio (P/E)",                  # 0.9 %
    "Target Total Return Year To Date Pct",               # 3.1 %
    "Target - Price Change 1 Year Percent (CHG_PCT_1Y)",  # 2.5 %
    "Target - Price Change 5 Day Percent (CHG_PCT_5D)",   # 3.1 %
}

# ─── Step 3: near-duplicate columns (keep the Trailing-12M variant) ─────────
DROP_DUPLICATE = {
    "Target Dividend Per Share",    # ≈ Target Trailg 12Mth Dividend per Shar
    "Acquirer Dividend Per Share",  # ≈ Acquirer Trailg 12Mth Dividend per Shar
}

ALL_DROP = DROP_CONSTANT | DROP_SPARSE | DROP_DUPLICATE

# ─── Step 4: missing-value sentinels to normalise ──────────────────────────────
SENTINELS = {"n.a.", "n/a", "#n/a", "nan", "null", "none", "-", "--", "inf", "-inf", "#value!"}

# ─── Step 5: columns that should remain as text (NOT parsed as float) ──────────
TEXT_COLS = {
    "Announce Date",
    "Target Name",
    "Acquirer Name",
    "Target Ticker",
    "Acquirer Ticker",
    "Currency of Deal",
    "Payment Type",
    "Deal Attributes",
    "Current Target SIC Code",
    "Current Acquirer SIC Code",
}

# ─── Step 7: payment-type dummy helpers ─────────────────────────────────────────
def payment_dummies(pay_type: str) -> dict:
    """Return binary dummy dict for a Payment Type string."""
    pt = pay_type.lower()
    return {
        "Payment_Cash":  "1" if "cash" in pt else "0",
        "Payment_Stock": "1" if "stock" in pt else "0",
        "Payment_Debt":  "1" if "debt" in pt else "0",
    }

DUMMY_COLS = ["Payment_Cash", "Payment_Stock", "Payment_Debt"]

# ─── helpers ────────────────────────────────────────────────────────────────────
def clean_numeric(val: str) -> str:
    """Strip whitespace/commas/currency symbols and try float(); return '' on fail."""
    v = val.strip().replace(",", "").replace("$", "").replace("£", "").replace("€", "")
    if not v or v.lower() in SENTINELS:
        return ""
    try:
        float(v)
        return v
    except ValueError:
        return ""


def clean_date(val: str) -> str:
    """Convert YYYY/M/D → YYYY-MM-DD."""
    v = val.strip()
    if not v:
        return ""
    try:
        dt = datetime.strptime(v, "%Y/%m/%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        # try other common formats
        for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(v, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return v  # return as-is if no format matched


# ─── main pipeline ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("M&A Data Cleaning Pipeline")
    print("=" * 70)

    # ── Read input ──────────────────────────────────────────────────────────
    with open(INPUT, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        in_headers = list(reader.fieldnames or [])
        rows = list(reader)

    print(f"\nInput : {len(rows)} rows × {len(in_headers)} cols")

    # ── Determine output columns ────────────────────────────────────────────
    kept_cols = [h for h in in_headers if h not in ALL_DROP]
    # Insert payment dummies right after Payment Type
    pt_idx = kept_cols.index("Payment Type") + 1
    for i, dc in enumerate(DUMMY_COLS):
        kept_cols.insert(pt_idx + i, dc)

    out_headers = kept_cols

    # ── Extract Deal Descriptions into separate file ─────────────────────
    nlp_headers = ["Announce Date", "Acquirer Ticker", "Target Ticker", "Deal Description"]
    nlp_rows = []

    # ── Process rows ────────────────────────────────────────────────────────
    seen_keys = set()
    cleaned_rows = []
    dup_count = 0

    for row in rows:
        # ── Step 8: de-duplicate on merge keys ──────────────────────────────
        key = (
            row.get("Announce Date", ""),
            row.get("Announced Total Value (mil.)", ""),
            row.get("Target Ticker", ""),
            row.get("Acquirer Ticker", ""),
        )
        if key in seen_keys:
            dup_count += 1
            continue
        seen_keys.add(key)

        # ── Build NLP row ───────────────────────────────────────────────────
        nlp_row = {
            "Announce Date":    clean_date(row.get("Announce Date", "")),
            "Acquirer Ticker":  row.get("Acquirer Ticker", "").strip(),
            "Target Ticker":    row.get("Target Ticker", "").strip(),
            "Deal Description": row.get("Deal Description", "").strip(),
        }
        nlp_rows.append(nlp_row)

        # ── Build cleaned row ───────────────────────────────────────────────
        out = {}
        for h in out_headers:
            if h in DUMMY_COLS:
                continue  # filled below
            raw = row.get(h, "").strip()

            # Step 4: normalise sentinels
            if raw.lower() in SENTINELS:
                raw = ""

            # Step 6: parse dates
            if h == "Announce Date":
                raw = clean_date(raw)

            # Step 5: clean numerics (only for non-text columns)
            elif h not in TEXT_COLS and raw:
                raw = clean_numeric(raw)

            out[h] = raw

        # Step 7: payment dummies
        dummies = payment_dummies(row.get("Payment Type", ""))
        out.update(dummies)

        cleaned_rows.append(out)

    # ── Write cleaned CSV ───────────────────────────────────────────────────
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_headers)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    # ── Write NLP descriptions CSV ──────────────────────────────────────────
    with open(NLP_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=nlp_headers)
        writer.writeheader()
        writer.writerows(nlp_rows)

    # ── Summary report ──────────────────────────────────────────────────────
    print(f"Output: {len(cleaned_rows)} rows × {len(out_headers)} cols")
    print(f"Duplicates removed: {dup_count}")
    print(f"\nColumns dropped ({len(ALL_DROP)}):")
    for c in sorted(ALL_DROP):
        print(f"  ✗ {c}")

    print(f"\nColumns retained ({len(out_headers)}):")
    for i, c in enumerate(out_headers, 1):
        # compute fill rate
        filled = sum(1 for r in cleaned_rows if r.get(c, "").strip())
        pct = 100.0 * filled / len(cleaned_rows) if cleaned_rows else 0
        print(f"  {i:>2}. {c:<55} {filled:>5}/{len(cleaned_rows)}  ({pct:.1f}%)")

    print(f"\n✅ Cleaned data  → {OUTPUT}")
    print(f"✅ NLP file      → {NLP_OUT}")

    # ── Quick sentinel check ────────────────────────────────────────────────
    sentinel_found = 0
    with open(OUTPUT, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for v in r.values():
                if v.strip().lower() in SENTINELS:
                    sentinel_found += 1
    if sentinel_found:
        print(f"\n⚠️  Sentinel values still present: {sentinel_found}")
    else:
        print(f"\n✅ No sentinel values (N.A., NaN, etc.) remaining")

    # ── Date format check ───────────────────────────────────────────────────
    bad_dates = 0
    iso_pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for r in cleaned_rows:
        d = r.get("Announce Date", "")
        if d and not iso_pat.match(d):
            bad_dates += 1
    if bad_dates:
        print(f"⚠️  Non-ISO dates found: {bad_dates}")
    else:
        print(f"✅ All dates in YYYY-MM-DD format")

    print("=" * 70)


if __name__ == "__main__":
    main()
