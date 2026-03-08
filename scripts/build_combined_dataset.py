#!/usr/bin/env python3
"""
Merge financial features (ma_cleaned.csv) with FinBERT text embeddings
(text_embeddings.csv) to create the combined dataset.

Usage:
    python scripts/build_combined_dataset.py
    python scripts/build_combined_dataset.py --output data/processed/combined.csv
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_combined_dataset(
    financial_path: Path,
    text_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Left-join financial features with text embeddings on (ticker, date).

    Parameters
    ----------
    financial_path : Path
        Path to ma_cleaned.csv.
    text_path : Path
        Path to text_embeddings.csv.
    output_path : Path
        Where to save combined CSV.

    Returns
    -------
    pd.DataFrame
        Combined dataset.
    """
    # ── Load financial data ──────────────────────────────────────────
    print("Loading financial data...")
    fin = pd.read_csv(financial_path)
    print(f"  Financial: {len(fin)} deals, {fin.shape[1]} columns")

    # Create join key matching the EDGAR pipeline format: "TICKER|DATE"
    fin["deal_key"] = (
        fin["Acquirer Ticker"].astype(str).str.strip()
        + "|"
        + fin["Announce Date"].astype(str).str.strip()
    )

    # ── Load text embeddings ─────────────────────────────────────────
    print("Loading text embeddings...")
    text = pd.read_csv(text_path)
    print(f"  Text: {len(text)} deals, {text.shape[1]} columns")

    # Drop redundant columns from text (keep only deal_key + features)
    text_features = text.drop(columns=["ticker", "announce_date"], errors="ignore")

    # Deduplicate: same ticker+date can appear if the company had multiple deals
    n_before = len(text_features)
    text_features = text_features.drop_duplicates(subset=["deal_key"], keep="first")
    n_dupes = n_before - len(text_features)
    if n_dupes:
        print(f"  Deduplicated: {n_dupes} duplicate text entries removed")

    # ── Merge ────────────────────────────────────────────────────────
    print("Merging on deal_key...")
    combined = fin.merge(text_features, on="deal_key", how="left")

    # Stats
    has_text = combined["has_mda"].fillna(False).astype(bool)
    n_with_text = has_text.sum()
    n_total = len(combined)
    pct = 100 * n_with_text / n_total

    print(f"\n  Combined: {n_total} deals, {combined.shape[1]} columns")
    print(f"  With text embeddings: {n_with_text} ({pct:.1f}%)")
    print(f"  Without text: {n_total - n_with_text} ({100 - pct:.1f}%)")

    # ── Fill NaN embeddings with zeros for deals without text ─────────
    mda_cols = [c for c in combined.columns if c.startswith("mda_pca_")]
    rf_cols = [c for c in combined.columns if c.startswith("rf_pca_")]
    combined[mda_cols + rf_cols] = combined[mda_cols + rf_cols].fillna(0.0)
    combined["has_mda"] = combined["has_mda"].fillna(False)
    combined["has_rf"] = combined["has_rf"].fillna(False)

    # ── Save ─────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\n✅ Combined dataset saved → {output_path}")
    print(f"   Shape: {combined.shape[0]} rows × {combined.shape[1]} columns")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Merge financial + text features into combined dataset",
    )
    parser.add_argument(
        "--financial", type=str,
        default=str(PROJECT_ROOT / "data" / "interim" / "ma_cleaned.csv"),
        help="Path to financial data CSV",
    )
    parser.add_argument(
        "--text", type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "text_embeddings.csv"),
        help="Path to text embeddings CSV",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "combined_financial_text.csv"),
        help="Output path for combined dataset",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Building Combined Financial + Text Dataset")
    print("=" * 60)

    combined = build_combined_dataset(
        financial_path=Path(args.financial),
        text_path=Path(args.text),
        output_path=Path(args.output),
    )

    # Summary stats
    print("\n" + "-" * 40)
    print("Column groups:")
    print(f"  Financial features: {len([c for c in combined.columns if not c.startswith(('mda_', 'rf_', 'has_', 'deal_key'))])} columns")
    print(f"  MD&A embeddings: {len([c for c in combined.columns if c.startswith('mda_pca_')])} dims")
    print(f"  Risk Factor embeddings: {len([c for c in combined.columns if c.startswith('rf_pca_')])} dims")
    print(f"  Metadata: deal_key, has_mda, has_rf")


if __name__ == "__main__":
    main()
