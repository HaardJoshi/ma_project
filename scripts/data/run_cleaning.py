#!/usr/bin/env python3
"""Clean combined CSV → data/interim/ma_cleaned.csv + ma_deal_descriptions.csv"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cleaning import clean

if __name__ == "__main__":
    clean()
