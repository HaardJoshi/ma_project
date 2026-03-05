"""
Preprocessing — winsorisation, z-score standardisation, train/val/test split.

Reads ``data/interim/ma_cleaned.csv`` and writes feature matrices to
``data/processed/``.
"""

import csv
import math
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ─── Columns that are identifiers / text, NOT features ─────────────────────────
ID_COLS = {
    "Announce Date", "Target Name", "Acquirer Name",
    "Target Ticker", "Acquirer Ticker", "Currency of Deal",
    "Payment Type", "Deal Attributes",
    "Current Target SIC Code", "Current Acquirer SIC Code",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    return headers, rows


def _to_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _winsorise(values: list[float | None], pct: float) -> list[float | None]:
    """Clip non-None values at the *pct* and *(1 - pct)* quantiles."""
    valid = sorted(v for v in values if v is not None)
    if len(valid) < 2:
        return values
    lo = valid[max(0, int(len(valid) * pct))]
    hi = valid[min(len(valid) - 1, int(len(valid) * (1 - pct)))]
    return [
        (None if v is None else max(lo, min(hi, v)))
        for v in values
    ]


def _zscore(values: list[float | None]) -> tuple[list[float | None], float, float]:
    """Z-score normalise; return (normalised_values, mean, std)."""
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return values, 0.0, 1.0
    mean = sum(valid) / len(valid)
    std = math.sqrt(sum((v - mean) ** 2 for v in valid) / (len(valid) - 1))
    if std == 0:
        std = 1.0
    return [
        (None if v is None else (v - mean) / std)
        for v in values
    ], mean, std


def preprocess(cfg: dict) -> dict[str, str]:
    """
    Run preprocessing: winsorise, z-score, split.

    Parameters
    ----------
    cfg : dict
        Loaded YAML config.

    Returns
    -------
    dict[str, str]
        Paths to train/val/test CSVs.
    """
    input_path = Path(cfg["data"]["cleaned_file"])
    output_dir = Path(cfg["data"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    pct = cfg["preprocessing"]["winsorise_percentile"]
    test_frac = cfg["preprocessing"]["test_size"]
    val_frac = cfg["preprocessing"]["val_size"]
    seed = cfg["project"]["seed"]

    headers, rows = _read_csv(input_path)
    numeric_cols = [h for h in headers if h not in ID_COLS]

    print(f"Preprocessing {len(rows)} rows, {len(numeric_cols)} numeric features")

    # ── 1. Parse & winsorise & z-score each numeric column ──────────────
    col_values: dict[str, list[float | None]] = {}
    for col in numeric_cols:
        raw = [_to_float(r.get(col, "")) for r in rows]
        winsorised = _winsorise(raw, pct)
        normalised, mean, std = _zscore(winsorised)
        col_values[col] = normalised

    # ── 2. Write back into rows ─────────────────────────────────────────
    for i, row in enumerate(rows):
        for col in numeric_cols:
            v = col_values[col][i]
            row[col] = f"{v:.6f}" if v is not None else ""

    # ── 3. Shuffle & split ──────────────────────────────────────────────
    random.seed(seed)
    indices = list(range(len(rows)))
    random.shuffle(indices)

    n_test = int(len(rows) * test_frac)
    n_val = int(len(rows) * val_frac)

    test_idx = set(indices[:n_test])
    val_idx = set(indices[n_test : n_test + n_val])
    train_idx = set(indices[n_test + n_val :])

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    output_paths = {}

    for split_name, idx_set in splits.items():
        split_rows = [rows[i] for i in sorted(idx_set)]
        out_path = output_dir / f"{split_name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(split_rows)
        output_paths[split_name] = str(out_path)
        print(f"  {split_name}: {len(split_rows)} rows → {out_path}")

    print("✅ Preprocessing complete")
    return output_paths


if __name__ == "__main__":
    from src.config import load_config
    preprocess(load_config())
