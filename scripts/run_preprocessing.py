#!/usr/bin/env python3
"""Preprocess cleaned CSV → data/processed/{train,val,test}.csv"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.preprocessing import preprocess


def main():
    parser = argparse.ArgumentParser(description="Preprocess M&A data")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    preprocess(cfg)


if __name__ == "__main__":
    main()
