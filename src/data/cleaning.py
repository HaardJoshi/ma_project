"""
Data cleaning pipeline.

Reads ``data/interim/ma_combined.csv`` and produces:
- ``data/interim/ma_cleaned.csv``          – cleaned tabular features
- ``data/interim/ma_deal_descriptions.csv`` – deal descriptions for NLP
"""

import csv
import os
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── Columns to drop ───────────────────────────────────────────────────────────
DROP_CONSTANT = {
    "Deal Type", "Deal Status", "Deal Description",
    "Legal Adviser to Target Financial",
}
DROP_SPARSE = {
    "Seller Name", "Seller Ticker", "Target Current Market Cap",
    "Target Price Earnings Ratio (P/E)", "Target Total Return Year To Date Pct",
    "Target - Price Change 1 Year Percent (CHG_PCT_1Y)",
    "Target - Price Change 5 Day Percent (CHG_PCT_5D)",
}
DROP_DUPLICATE = {"Target Dividend Per Share", "Acquirer Dividend Per Share"}
ALL_DROP = DROP_CONSTANT | DROP_SPARSE | DROP_DUPLICATE

# ─── Sentinel values ───────────────────────────────────────────────────────────
SENTINELS = {"n.a.", "n/a", "#n/a", "nan", "null", "none", "-", "--", "inf", "-inf", "#value!"}

# ─── Text columns (not parsed as float) ────────────────────────────────────────
TEXT_COLS = {
    "Announce Date", "Target Name", "Acquirer Name", "Target Ticker",
    "Acquirer Ticker", "Currency of Deal", "Payment Type", "Deal Attributes",
    "Current Target SIC Code", "Current Acquirer SIC Code",
}

DUMMY_COLS = ["Payment_Cash", "Payment_Stock", "Payment_Debt"]


def _payment_dummies(pay_type: str) -> dict:
    pt = pay_type.lower()
    return {
        "Payment_Cash":  "1" if "cash" in pt else "0",
        "Payment_Stock": "1" if "stock" in pt else "0",
        "Payment_Debt":  "1" if "debt" in pt else "0",
    }


def _clean_numeric(val: str) -> str:
    v = val.strip().replace(",", "").replace("$", "").replace("£", "").replace("€", "")
    if not v or v.lower() in SENTINELS:
        return ""
    try:
        float(v)
        return v
    except ValueError:
        return ""


def _clean_date(val: str) -> str:
    v = val.strip()
    if not v:
        return ""
    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return v


def clean(
    input_path: str | None = None,
    output_path: str | None = None,
    nlp_path: str | None = None,
) -> tuple[str, str]:
    """
    Run the full cleaning pipeline.

    Returns
    -------
    tuple[str, str]
        Paths to (cleaned_csv, nlp_csv).
    """
    interim = PROJECT_ROOT / "data" / "interim"
    input_path = Path(input_path) if input_path else interim / "ma_combined.csv"
    output_path = Path(output_path) if output_path else interim / "ma_cleaned.csv"
    nlp_path = Path(nlp_path) if nlp_path else interim / "ma_deal_descriptions.csv"

    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        in_headers = list(reader.fieldnames or [])
        rows = list(reader)

    print(f"Input: {len(rows)} rows × {len(in_headers)} cols")

    kept_cols = [h for h in in_headers if h not in ALL_DROP]
    pt_idx = kept_cols.index("Payment Type") + 1
    for i, dc in enumerate(DUMMY_COLS):
        kept_cols.insert(pt_idx + i, dc)
    out_headers = kept_cols

    nlp_headers = ["Announce Date", "Acquirer Ticker", "Target Ticker", "Deal Description"]
    nlp_rows = []
    seen_keys = set()
    cleaned_rows = []

    for row in rows:
        key = (row.get("Announce Date", ""), row.get("Announced Total Value (mil.)", ""),
               row.get("Target Ticker", ""), row.get("Acquirer Ticker", ""))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        nlp_rows.append({
            "Announce Date": _clean_date(row.get("Announce Date", "")),
            "Acquirer Ticker": row.get("Acquirer Ticker", "").strip(),
            "Target Ticker": row.get("Target Ticker", "").strip(),
            "Deal Description": row.get("Deal Description", "").strip(),
        })

        out = {}
        for h in out_headers:
            if h in DUMMY_COLS:
                continue
            raw = row.get(h, "").strip()
            if raw.lower() in SENTINELS:
                raw = ""
            if h == "Announce Date":
                raw = _clean_date(raw)
            elif h not in TEXT_COLS and raw:
                raw = _clean_numeric(raw)
            out[h] = raw
        out.update(_payment_dummies(row.get("Payment Type", "")))
        cleaned_rows.append(out)

    for path, headers, data in [(output_path, out_headers, cleaned_rows),
                                 (nlp_path, nlp_headers, nlp_rows)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)

    print(f"✅ Cleaned  → {output_path}  ({len(cleaned_rows)} rows × {len(out_headers)} cols)")
    print(f"✅ NLP file → {nlp_path}  ({len(nlp_rows)} rows × {len(nlp_headers)} cols)")
    return str(output_path), str(nlp_path)


if __name__ == "__main__":
    clean()
