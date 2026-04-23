#!/usr/bin/env python3
"""
Extract FinBERT text embeddings from EDGAR 10-K filings.

Usage:
    # Test with 5 deals
    python scripts/run_text_features.py --limit 5

    # Full pipeline with custom PCA dims
    python scripts/run_text_features.py --pca-dims 32

    # Skip PCA (save raw 768-dim embeddings only)
    python scripts/run_text_features.py --no-pca
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.text import (
    FinBERTEmbedder,
    build_text_embeddings,
    reduce_dimensions,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract FinBERT embeddings from 10-K filings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pca-dims", type=int, default=64,
        help="Number of PCA components per section (default: 64, total: 128)",
    )
    parser.add_argument(
        "--no-pca", action="store_true",
        help="Skip PCA, save raw 768-dim embeddings only",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of deals to process (for testing)",
    )
    parser.add_argument(
        "--model", type=str, default="ProsusAI/finbert",
        help="HuggingFace model name (default: ProsusAI/finbert)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda', 'mps', or 'cpu' (auto-detected if omitted)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("FinBERT Text Embedding Pipeline")
    print("=" * 60)

    # Stage 1: Load model
    embedder = FinBERTEmbedder(
        model_name=args.model,
        device=args.device,
    )

    # Stage 2: Extract raw embeddings
    print("\n" + "-" * 40)
    print("Extracting raw embeddings...")
    print("-" * 40)

    mda_emb, rf_emb, metadata = build_text_embeddings(
        embedder=embedder,
        limit=args.limit,
    )

    # Stage 3: PCA reduction
    if not args.no_pca:
        print("\n" + "-" * 40)
        print(f"PCA reduction (768 → {args.pca_dims} per section)...")
        print("-" * 40)

        output_csv = reduce_dimensions(
            mda_emb, rf_emb, metadata,
            n_components=args.pca_dims,
        )
        print(f"\n✅ Pipeline complete. Output: {output_csv}")
    else:
        print("\n✅ Raw embeddings saved (PCA skipped).")


if __name__ == "__main__":
    main()
