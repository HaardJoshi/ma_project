"""
Combine 5 M&A CSV exports into a single file.

Uses only the Python standard library (csv module) — no pandas required.
Merge logic:
  • All five files share 4 key columns used to match rows across files.
  • Each file contributes its own unique (non-key) columns.
  • When a column appears in more than one file, the value from the
    *first* file it was seen in is kept (no duplicates).
"""

import csv
import os
from collections import OrderedDict

# ─── Configuration ──────────────────────────────────────────────────────────────
INPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(INPUT_DIR, "ma_combined.csv")

# Process the richest metadata file first
CSV_FILES = [
    "ma_export_33205636_172324.csv",
    "ma_export_33205636_161332.csv",
    "ma_export_33205636_162411.csv",
    "ma_export_33205636_175852.csv",
    "ma_export_33205636_182631.csv",
]

MERGE_KEYS = ["Announce Date", "Announced Total Value (mil.)", "Target Ticker", "Acquirer Ticker"]


def make_key(row: dict) -> tuple:
    """Create a hashable merge key from a row dict."""
    return tuple(row.get(k, "") for k in MERGE_KEYS)


def main():
    print("=" * 70)
    print("M&A Data Merge – combining 5 exports into one file")
    print("=" * 70)

    # Will hold: merge_key -> OrderedDict of {column_name: value}
    combined: dict[tuple, OrderedDict] = {}
    # Track the final ordered list of all columns
    all_columns: list[str] = []
    seen_columns: set[str] = set()

    for fname in CSV_FILES:
        path = os.path.join(INPUT_DIR, fname)
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            file_headers = reader.fieldnames or []

            # Determine which columns are new from this file
            new_cols = [c for c in file_headers if c not in seen_columns]
            overlap_cols = [c for c in file_headers if c in seen_columns and c not in MERGE_KEYS]

            print(f"\n  {fname}")
            print(f"    Columns in file: {len(file_headers)}")
            print(f"    New columns added: {len(new_cols)}  {new_cols}")
            if overlap_cols:
                print(f"    Overlapping (kept from earlier): {overlap_cols}")

            # Register new columns in order
            for c in new_cols:
                all_columns.append(c)
                seen_columns.add(c)

            row_count = 0
            for row in reader:
                key = make_key(row)
                row_count += 1

                if key not in combined:
                    # First time seeing this deal – initialise with empty strings
                    combined[key] = OrderedDict((c, "") for c in all_columns)

                # Fill in values only for NEW columns from this file
                for c in new_cols:
                    combined[key][c] = row.get(c, "")

                # Also ensure any new columns added by THIS file get empty
                # strings for deals that were already seen from earlier files.
                # (Handled by the default "" above in OrderedDict init for
                #  future keys; for existing keys we patch below.)

            # Patch existing rows that predate this file: add missing new cols
            for key, data in combined.items():
                for c in new_cols:
                    if c not in data:
                        data[c] = ""

            print(f"    Rows read: {row_count}")

    # ─── Write output ───────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for data in combined.values():
            # Ensure every column is present (fill missing with "")
            row_out = {c: data.get(c, "") for c in all_columns}
            writer.writerow(row_out)

    total_rows = len(combined)
    print("\n" + "=" * 70)
    print(f"✅ Combined file saved → {OUTPUT_FILE}")
    print(f"   Final shape: {total_rows} rows × {len(all_columns)} cols")
    print(f"\n   All {len(all_columns)} columns:")
    for i, col in enumerate(all_columns, 1):
        print(f"     {i:>2}. {col}")
    print("=" * 70)


if __name__ == "__main__":
    main()
