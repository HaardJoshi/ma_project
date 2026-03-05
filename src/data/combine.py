"""
Combine multiple LSEG M&A CSV exports into a single file.

Reads all ``ma_export_*.csv`` files from ``data/raw/`` and merges them
on the shared key columns, adding only unique columns from each file.
Output is written to ``data/interim/ma_combined.csv``.
"""

import csv
import os
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MERGE_KEYS = [
    "Announce Date",
    "Announced Total Value (mil.)",
    "Target Ticker",
    "Acquirer Ticker",
]


def make_key(row: dict) -> tuple:
    return tuple(row.get(k, "") for k in MERGE_KEYS)


def combine(raw_dir: str | None = None, output_path: str | None = None) -> str:
    """
    Combine all ``ma_export_*.csv`` files in *raw_dir*.

    Parameters
    ----------
    raw_dir : str, optional
        Directory containing the raw exports. Defaults to ``data/raw/``.
    output_path : str, optional
        Where to write the combined CSV. Defaults to ``data/interim/ma_combined.csv``.

    Returns
    -------
    str
        Absolute path to the combined CSV.
    """
    raw_dir = Path(raw_dir) if raw_dir else PROJECT_ROOT / "data" / "raw"
    output_path = Path(output_path) if output_path else PROJECT_ROOT / "data" / "interim" / "ma_combined.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("ma_export_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No ma_export_*.csv files found in {raw_dir}")

    print(f"Found {len(csv_files)} CSV exports in {raw_dir}")

    combined: dict[tuple, OrderedDict] = {}
    all_columns: list[str] = []
    seen_columns: set[str] = set()

    for fpath in csv_files:
        with open(fpath, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            file_headers = reader.fieldnames or []
            new_cols = [c for c in file_headers if c not in seen_columns]

            for c in new_cols:
                all_columns.append(c)
                seen_columns.add(c)

            row_count = 0
            for row in reader:
                key = make_key(row)
                row_count += 1
                if key not in combined:
                    combined[key] = OrderedDict((c, "") for c in all_columns)
                for c in new_cols:
                    combined[key][c] = row.get(c, "")

            for key, data in combined.items():
                for c in new_cols:
                    if c not in data:
                        data[c] = ""

            print(f"  {fpath.name}: {row_count} rows, {len(new_cols)} new cols")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for data in combined.values():
            writer.writerow({c: data.get(c, "") for c in all_columns})

    print(f"✅ Combined → {output_path}  ({len(combined)} rows × {len(all_columns)} cols)")
    return str(output_path)


if __name__ == "__main__":
    combine()
