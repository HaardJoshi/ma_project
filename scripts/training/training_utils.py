"""
training_utils.py  --  Shared data loading, preprocessing, and evaluation
================================================================================
Used by all model training scripts (train_ridge.py, train_xgboost.py, etc.)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

SEED = 42
N_FOLDS = 5
DATA_FILE = "data/processed/final_multimodal_dataset.csv"
RESULTS_DIR = "results"
TARGET_COL = "car_m5_p5"

# ── Feature column groups ───────────────────────────────────────────────────
FINANCIAL_COLS = [
    "Announced Total Value (mil.)", "Payment_Cash", "Payment_Stock", "Payment_Debt",
    "TV/EBITDA", "Target Total Assets", "Acquirer Total Assets",
    "Acquirer Current Market Cap", "Acquirer Price Earnings Ratio (P/E)",
    "Target Trailg 12Mth Dividend per Shar", "Acquirer Trailg 12Mth Dividend per Shar",
    "Target Dividend Payout Ratio", "Acquirer Dividend Payout Ratio",
    "Target Market Value of Equity", "Acquirer Market Value of Equity",
    "Acquirer Total Return Year To Date Pct",
    "Target Trailg 12Mth Cashflow Net Inc", "Acquirer Trailg 12Mth Cashflow Net Inc",
    "Target R & D Expenditures", "Acquirer R & D Expenditures",
    "Target Inventories", "Acquirer Inventories",
    "Acquirer - Price Change 1 Year Percent (CHG_PCT_1Y)",
    "Acquirer - Price Change 5 Day Percent (CHG_PCT_5D)",
    "Acquirer Trailing 12 Mth COGS", "Target Trailing 12 Mth COGS",
    "Acquirer Trailing 12 month EBITDA per Share",
    "Target Trailing 12 month EBITDA per Share",
    "Target Sales/Revenue/Turnover", "Acquirer Sales/Revenue/Turnover",
    "Target Return on Common Equity", "Acquirer Return on Common Equity",
    "Target EBITDA(Earn Bef Int Dep & Amo)", "Acquirer EBITDA(Earn Bef Int Dep & Amo)",
    "Target Operating Margin", "Acquirer Operating Margin",
    "Target Total Debt to Total Assets", "Acquirer Total Debt to Total Assets",
    "Target Total Debt to Total Equity", "Acquirer Total Debt to Total Equity",
    "Target EBIT to Total Interest Expense", "Acquirer EBIT to Total Interest Expense",
    "Target Current Ratio", "Acquirer Current Ratio",
    "Target Financial Leverage", "Acquirer Financial Leverage",
    "Target GeoGrwth - Cash Flow per Share", "Acquirer GeoGrwth - Cash Flow per Share",
    "Target Net Income/Net Profit (Losses)", "Acquirer Net Income/Net Profit (Losses)",
    "Target Net Revenue Growth", "Acquirer Net Revenue Growth",
    "Target Asset Growth", "Acquirer Asset Growth",
    "Target Geometric Growth-EBITDA Tot Mkt VaL",
    "Acquirer Geometric Growth-EBITDA Tot Mkt VaL",
]

TEXT_COLS = [f"mda_pca_{i}" for i in range(64)] + [f"rf_pca_{i}" for i in range(64)]
GRAPH_COLS = [f"graph_emb_{i}" for i in range(64)]


def get_feature_configs(df):
    """Build feature configs with only columns that exist in the data."""
    available_fin = [c for c in FINANCIAL_COLS if c in df.columns]
    available_text = [c for c in TEXT_COLS if c in df.columns]
    available_graph = [c for c in GRAPH_COLS if c in df.columns]

    return {
        "M1": {"name": "Financial Only", "cols": available_fin},
        "M2": {"name": "Financial + Text", "cols": available_fin + available_text},
        "M3": {"name": "Financial + Text + Graph", "cols": available_fin + available_text + available_graph},
    }


def winsorize(series, lower=0.01, upper=0.99):
    """Cap extreme values at given percentiles."""
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lo, hi)


def load_and_prepare_data():
    """Load data, filter to valid subset, winsorize."""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    # Filter to deals with CAR + Graph + minimum financials
    has_car = df[TARGET_COL].notna()
    has_graph = df["has_graph"] == 1 if "has_graph" in df.columns else (
        df[GRAPH_COLS].abs().sum(axis=1) > 0
    )
    available_fin = [c for c in FINANCIAL_COLS if c in df.columns]
    has_financials = df[available_fin].notna().sum(axis=1) > (len(available_fin) * 0.5)

    subset = df[has_car & has_graph & has_financials].copy()
    print(f"  Full dataset:  {len(df):,} deals")
    print(f"  Filtered:      {len(subset):,} deals (CAR + Graph + Financials)")

    # Winsorize financial features and target
    print("  Winsorizing at 1st/99th percentile...")
    for col in available_fin:
        if col in subset.columns:
            subset[col] = winsorize(subset[col])
    subset[TARGET_COL] = winsorize(subset[TARGET_COL])

    y = subset[TARGET_COL].values
    print(f"  Target (CAR): mean={y.mean():.4f}, std={y.std():.4f}")

    return subset, y


def compute_metrics(y_true, y_pred):
    """Compute all regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    residuals = y_true - y_pred
    huber = np.mean(np.where(
        np.abs(residuals) <= 1.0,
        0.5 * residuals**2,
        1.0 * (np.abs(residuals) - 0.5)
    ))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Huber": huber}


def run_cv(model_builder, X, y, model_name, config_name, n_folds=N_FOLDS):
    """Run k-fold cross-validation for a given model builder."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        fold_metrics.append(metrics)

        print(f"    Fold {fold_idx+1}/{n_folds}: R²={metrics['R2']:.4f}, "
              f"RMSE={metrics['RMSE']:.4f}")

    # Aggregate
    avg = {m: np.mean([f[m] for f in fold_metrics]) for m in fold_metrics[0]}
    std = {m: np.std([f[m] for f in fold_metrics]) for m in fold_metrics[0]}
    fold_r2s = [f["R2"] for f in fold_metrics]

    print(f"  ► Mean R²={avg['R2']:.4f}±{std['R2']:.4f} | "
          f"RMSE={avg['RMSE']:.4f}±{std['RMSE']:.4f} | MAE={avg['MAE']:.4f}")

    return {
        "config": config_name,
        "model": model_name,
        "n_features": X.shape[1],
        "R2_mean": avg["R2"], "R2_std": std["R2"],
        "RMSE_mean": avg["RMSE"], "RMSE_std": std["RMSE"],
        "MAE_mean": avg["MAE"], "MAE_std": std["MAE"],
        "MSE_mean": avg["MSE"], "Huber_mean": avg["Huber"],
        "fold_r2s": fold_r2s,
    }


def save_results(results, model_name):
    """Save results to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{RESULTS_DIR}/{model_name}_results_{timestamp}.csv"

    df = pd.DataFrame(results)
    save_cols = [c for c in df.columns if c != "fold_r2s"]
    df[save_cols].to_csv(filepath, index=False)
    print(f"\n✅ Results saved to {filepath}")
    return filepath


def print_significance(results):
    """Print paired t-test comparing M1 vs M3."""
    m1 = [r for r in results if r["config"] == "M1"]
    m3 = [r for r in results if r["config"] == "M3"]

    if m1 and m3:
        t_stat, p_value = stats.ttest_rel(m3[0]["fold_r2s"], m1[0]["fold_r2s"])
        sig = "✅ p<0.05" if p_value < 0.05 else "❌ p≥0.05"
        print(f"\n  M1 vs M3 (paired t-test): t={t_stat:.3f}, p={p_value:.4f} | {sig}")
