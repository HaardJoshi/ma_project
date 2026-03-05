#!/usr/bin/env python3
"""Combine raw CSV exports → data/interim/ma_combined.csv"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.combine import combine

if __name__ == "__main__":
    combine()
