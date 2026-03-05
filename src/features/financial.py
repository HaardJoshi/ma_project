"""
Block A — Financial feature extraction.

Reads the cleaned M&A CSV and produces a standardised numeric feature
vector h_F for each deal.  This module handles:
  - Selecting the relevant financial ratio columns
  - Gap-filling / imputation strategy
  - Producing the final feature matrix

The heavy preprocessing (winsorisation, z-score) is done upstream in
``src/data/preprocessing.py``; this module adds any deal-level derived
features (e.g. relative size, sector dummies).
"""

from pathlib import Path

# Financial ratio columns expected in the cleaned CSV
FINANCIAL_COLS = [
    # ── Deal-level ──────────────────────────────────────
    "Announced Total Value (mil.)",
    "TV/EBITDA",
    "Current/Completed Total Value",
    "Payment_Cash",
    "Payment_Stock",
    "Payment_Debt",
    # ── Target fundamentals ─────────────────────────────
    "Target Total Assets",
    "Target Market Value of Equity",
    "Target Sales/Revenue/Turnover",
    "Target EBITDA(Earn Bef Int Dep & Amo)",
    "Target Operating Margin",
    "Target Return on Common Equity",
    "Target Net Income/Net Profit (Losses)",
    "Target Total Debt to Total Assets",
    "Target Total Debt to Total Equity",
    "Target EBIT to Total Interest Expense",
    "Target Current Ratio",
    "Target Financial Leverage",
    "Target Trailg 12Mth Dividend per Shar",
    "Target Dividend Payout Ratio",
    "Target Trailg 12Mth Cashflow Net Inc",
    "Target Trailing 12 Mth COGS",
    "Target Trailing 12 month EBITDA per Share",
    "Target R & D Expenditures",
    "Target Inventories",
    "Target Net Revenue Growth",
    "Target Asset Growth",
    "Target GeoGrwth - Cash Flow per Share",
    "Target Geometric Growth-EBITDA Tot Mkt VaL",
    # ── Acquirer fundamentals ───────────────────────────
    "Acquirer Total Assets",
    "Acquirer Current Market Cap",
    "Acquirer Market Value of Equity",
    "Acquirer Sales/Revenue/Turnover",
    "Acquirer EBITDA(Earn Bef Int Dep & Amo)",
    "Acquirer Operating Margin",
    "Acquirer Return on Common Equity",
    "Acquirer Net Income/Net Profit (Losses)",
    "Acquirer Price Earnings Ratio (P/E)",
    "Acquirer Total Debt to Total Assets",
    "Acquirer Total Debt to Total Equity",
    "Acquirer EBIT to Total Interest Expense",
    "Acquirer Current Ratio",
    "Acquirer Financial Leverage",
    "Acquirer Trailg 12Mth Dividend per Shar",
    "Acquirer Dividend Payout Ratio",
    "Acquirer Trailg 12Mth Cashflow Net Inc",
    "Acquirer Trailing 12 Mth COGS",
    "Acquirer Trailing 12 month EBITDA per Share",
    "Acquirer R & D Expenditures",
    "Acquirer Inventories",
    "Acquirer Total Return Year To Date Pct",
    "Acquirer - Price Change 1 Year Percent (CHG_PCT_1Y)",
    "Acquirer - Price Change 5 Day Percent (CHG_PCT_5D)",
    "Acquirer Net Revenue Growth",
    "Acquirer Asset Growth",
    "Acquirer GeoGrwth - Cash Flow per Share",
    "Acquirer Geometric Growth-EBITDA Tot Mkt VaL",
]


def get_financial_feature_names() -> list[str]:
    """Return the ordered list of financial feature column names."""
    return list(FINANCIAL_COLS)


def compute_derived_features(row: dict) -> dict:
    """
    Compute additional derived features from raw columns.

    Parameters
    ----------
    row : dict
        A single deal record.

    Returns
    -------
    dict
        Derived feature key-value pairs (added to the feature vector).

    TODO
    ----
    - Relative deal size (deal value / acquirer market cap)
    - Acquirer-Target size ratio
    - Cross-sector flag (SIC code mismatch)
    """
    derived = {}

    # Example: relative deal size
    try:
        deal_val = float(row.get("Announced Total Value (mil.)", "") or 0)
        acq_cap = float(row.get("Acquirer Current Market Cap", "") or 0)
        derived["Relative_Deal_Size"] = deal_val / acq_cap if acq_cap > 0 else 0.0
    except (ValueError, TypeError):
        derived["Relative_Deal_Size"] = 0.0

    # Example: cross-sector flag
    target_sic = row.get("Current Target SIC Code", "").strip()
    acq_sic = row.get("Current Acquirer SIC Code", "").strip()
    derived["Cross_Sector"] = 1.0 if (target_sic and acq_sic and target_sic != acq_sic) else 0.0

    return derived
