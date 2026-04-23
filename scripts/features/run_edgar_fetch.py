#!/usr/bin/env python3
"""
Fetch SEC EDGAR 10-K filings for M&A deal acquirers.

Usage:
    # Test with 5 companies
    python scripts/run_edgar_fetch.py --email "Your Name your@email.com" --limit 5

    # Full pipeline
    python scripts/run_edgar_fetch.py --email "Your Name your@email.com"

    # Resume after interruption
    python scripts/run_edgar_fetch.py --email "Your Name your@email.com" --resume

    # Only run ticker → CIK mapping (Stage 1)
    python scripts/run_edgar_fetch.py --email "Your Name your@email.com" --stage mapping-only
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.edgar import run_pipeline, build_ticker_cik_map


def main():
    parser = argparse.ArgumentParser(
        description="Fetch SEC EDGAR 10-K filings for M&A acquirers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_edgar_fetch.py --email "Jane Doe jane@uni.ac.uk" --limit 5
  python scripts/run_edgar_fetch.py --email "Jane Doe jane@uni.ac.uk" --resume
  python scripts/run_edgar_fetch.py --email "Jane Doe jane@uni.ac.uk" --stage mapping-only
        """,
    )
    parser.add_argument(
        "--email", required=True,
        help='SEC User-Agent string: "Your Name your@email.com"',
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of deals to process (for testing)",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume from checkpoint (default: True)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start from scratch, ignore existing checkpoint",
    )
    parser.add_argument(
        "--stage", choices=["all", "mapping-only"], default="all",
        help="Which pipeline stages to run",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    resume = not args.no_resume

    if args.stage == "mapping-only":
        print("Running Stage 1 only: Ticker → CIK mapping")
        mapping = build_ticker_cik_map(args.email)
        print(f"✅ Mapped {len(mapping)} tickers → CIKs")
        return

    stats = run_pipeline(
        user_agent=args.email,
        limit=args.limit,
        resume=resume,
    )

    print("\n✅ EDGAR pipeline complete.")


if __name__ == "__main__":
    main()
