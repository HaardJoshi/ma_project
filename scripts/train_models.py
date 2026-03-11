"""
train_models.py  --  M&A Synergy Prediction Training Pipeline
================================================================================
Trains and evaluates all model configurations on the 2,112-deal subset
(deals with CAR + Financials + SPLC graph data).

Models:
  Tier 1: Ridge, ElasticNet (linear baselines)
  Tier 2: XGBoost (primary model)
  Tier 3: MLP Fusion (secondary, per methodology §2.1.2)

Feature configs:
  M1: Financial only (67 features)
  M2: Financial + Text (195 features)
  M3: Financial + Text + Graph (259 features)

Evaluation: 5-fold CV, MSE/RMSE/MAE/R²

Usage:
    python scripts/train_models.py             # run all
    python scripts/train_models.py --models xgboost  # XGBoost only
    python scripts/train_models.py --configs M1      # financial only
"""

import argparse
import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_FILE = "data/processed/final_multimodal_dataset.csv"
RESULTS_DIR = "results"
SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

# Feature column groups
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
TARGET_COL = "car_m5_p5"

# Model configurations
FEATURE_CONFIGS = {
    "M1": {"name": "Financial Only", "cols": FINANCIAL_COLS},
    "M2": {"name": "Financial + Text", "cols": FINANCIAL_COLS + TEXT_COLS},
    "M3": {"name": "Financial + Text + Graph", "cols": FINANCIAL_COLS + TEXT_COLS + GRAPH_COLS},
}


# ── WINSORIZATION ───────────────────────────────────────────────────────────
def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize a series at the given percentiles."""
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lo, hi)


# ── METRICS ─────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Huber loss (delta=1.0)
    residuals = y_true - y_pred
    huber = np.mean(np.where(
        np.abs(residuals) <= 1.0,
        0.5 * residuals**2,
        1.0 * (np.abs(residuals) - 0.5)
    ))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Huber": huber}


# ── MODEL BUILDERS ──────────────────────────────────────────────────────────
def build_ridge(alpha=1.0):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha, random_state=SEED)),
    ])


def build_elasticnet(alpha=0.1, l1_ratio=0.5):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=SEED)),
    ])


def build_xgboost():
    from xgboost import XGBRegressor
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # XGBoost doesn't need scaling, but we include it for consistency
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,       # L1
            reg_lambda=1.0,      # L2
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )),
    ])


def build_mlp(input_dim):
    """Build MLP fusion model per methodology §2.1.2."""
    import torch
    import torch.nn as nn

    class FusionMLP(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return FusionMLP(input_dim)


def train_mlp(model, X_train, y_train, X_val, y_val, epochs=200, lr=1e-3, weight_decay=1e-4):
    """Train MLP with early stopping."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 20

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    model.load_state_dict(best_state)
    return model


def predict_mlp(model, X):
    """Get predictions from MLP."""
    import torch
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(X_t).cpu().numpy()
    return pred


# ── MAIN PIPELINE ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train M&A synergy models")
    parser.add_argument("--models", nargs="+", default=["ridge", "elasticnet", "xgboost", "mlp"],
                        choices=["ridge", "elasticnet", "xgboost", "mlp"])
    parser.add_argument("--configs", nargs="+", default=["M1", "M2", "M3"],
                        choices=["M1", "M2", "M3"])
    args = parser.parse_args()

    print("=" * 70)
    print("  M&A SYNERGY PREDICTION — MODEL TRAINING PIPELINE")
    print("=" * 70)

    # ── Load and filter data ─────────────────────────────────────
    print("\nLoading data...")
    df = pd.read_csv(DATA_FILE)

    # Filter to deals with CAR + Graph data (our 2,112 subset)
    has_car = df[TARGET_COL].notna()
    has_graph = df["has_graph"] == 1 if "has_graph" in df.columns else (
        df[GRAPH_COLS].abs().sum(axis=1) > 0
    )
    # Check which financial cols actually exist in the data
    available_fin = [c for c in FINANCIAL_COLS if c in df.columns]
    has_financials = df[available_fin].notna().sum(axis=1) > (len(available_fin) * 0.5)

    subset = df[has_car & has_graph & has_financials].copy()
    print(f"  Full dataset: {len(df):,} deals")
    print(f"  Filtered (CAR + Graph + Financials): {len(subset):,} deals")

    # ── Winsorize ────────────────────────────────────────────────
    print("\nWinsorizing financial features and target (1st/99th percentile)...")
    for col in available_fin:
        if col in subset.columns:
            subset[col] = winsorize(subset[col])
    subset[TARGET_COL] = winsorize(subset[TARGET_COL])

    y = subset[TARGET_COL].values
    print(f"  Target (CAR) stats: mean={y.mean():.4f}, std={y.std():.4f}, "
          f"min={y.min():.4f}, max={y.max():.4f}")

    # Update FEATURE_CONFIGS with available columns
    FEATURE_CONFIGS["M1"]["cols"] = available_fin
    FEATURE_CONFIGS["M2"]["cols"] = available_fin + [c for c in TEXT_COLS if c in subset.columns]
    FEATURE_CONFIGS["M3"]["cols"] = available_fin + [c for c in TEXT_COLS if c in subset.columns] + [c for c in GRAPH_COLS if c in subset.columns]

    # ── Cross-validation ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CROSS-VALIDATION ({N_FOLDS}-fold)")
    print(f"{'='*70}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_results = []

    for config_name in args.configs:
        cfg = FEATURE_CONFIGS[config_name]
        feature_cols = cfg["cols"]
        X = subset[feature_cols].values

        print(f"\n{'─'*70}")
        print(f"  Config: {config_name} — {cfg['name']} ({len(feature_cols)} features)")
        print(f"{'─'*70}")

        for model_name in args.models:
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if model_name == "mlp":
                    # MLP needs its own preprocessing
                    imputer = SimpleImputer(strategy="median")
                    scaler = StandardScaler()
                    X_train_p = scaler.fit_transform(imputer.fit_transform(X_train))
                    X_test_p = scaler.transform(imputer.transform(X_test))

                    model = build_mlp(X_train_p.shape[1])
                    model = train_mlp(model, X_train_p, y_train, X_test_p, y_test)
                    y_pred = predict_mlp(model, X_test_p)
                else:
                    if model_name == "ridge":
                        model = build_ridge()
                    elif model_name == "elasticnet":
                        model = build_elasticnet()
                    elif model_name == "xgboost":
                        model = build_xgboost()

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                metrics = compute_metrics(y_test, y_pred)
                fold_metrics.append(metrics)

            # Aggregate across folds
            avg_metrics = {}
            std_metrics = {}
            for metric in fold_metrics[0].keys():
                vals = [fm[metric] for fm in fold_metrics]
                avg_metrics[metric] = np.mean(vals)
                std_metrics[metric] = np.std(vals)

            result = {
                "config": config_name,
                "config_name": cfg["name"],
                "model": model_name,
                "n_features": len(feature_cols),
            }
            for metric in avg_metrics:
                result[f"{metric}_mean"] = avg_metrics[metric]
                result[f"{metric}_std"] = std_metrics[metric]
            result["fold_r2s"] = [fm["R2"] for fm in fold_metrics]

            all_results.append(result)

            print(f"  {model_name:12s} | R²={avg_metrics['R2']:.4f}±{std_metrics['R2']:.4f} | "
                  f"RMSE={avg_metrics['RMSE']:.4f}±{std_metrics['RMSE']:.4f} | "
                  f"MAE={avg_metrics['MAE']:.4f}")

    # ── Results Summary ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")

    results_df = pd.DataFrame(all_results)
    summary_cols = ["config", "model", "n_features", "R2_mean", "R2_std",
                    "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "Huber_mean"]
    print(results_df[summary_cols].to_string(index=False))

    # ── Statistical significance (M1 vs M3 for each model) ──────
    print(f"\n{'─'*70}")
    print("  STATISTICAL SIGNIFICANCE (M1 vs M3, paired t-test)")
    print(f"{'─'*70}")

    for model_name in args.models:
        m1_result = [r for r in all_results if r["config"] == "M1" and r["model"] == model_name]
        m3_result = [r for r in all_results if r["config"] == "M3" and r["model"] == model_name]

        if m1_result and m3_result:
            m1_r2s = m1_result[0]["fold_r2s"]
            m3_r2s = m3_result[0]["fold_r2s"]
            t_stat, p_value = stats.ttest_rel(m3_r2s, m1_r2s)
            sig = "✅ p<0.05" if p_value < 0.05 else "❌ p≥0.05"
            print(f"  {model_name:12s} | t={t_stat:.3f}, p={p_value:.4f} | {sig}")

    # ── Save results ─────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{RESULTS_DIR}/training_results_{timestamp}.csv"

    # Remove fold_r2s (list) before saving
    save_df = results_df.drop(columns=["fold_r2s"], errors="ignore")
    save_df.to_csv(results_file, index=False)
    print(f"\n✅ Results saved to {results_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
