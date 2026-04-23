"""
train_ridge.py  --  Ridge & ElasticNet Linear Baselines
================================================================================
Trains Ridge and ElasticNet regression across all feature configs (M1, M2, M3).
These are the linear baselines — if they match XGBoost, the relationship is linear.

Usage:
    python scripts/train_ridge.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from training_utils import (
    load_and_prepare_data, get_feature_configs, run_cv,
    save_results, print_significance, SEED
)


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


def main():
    print("=" * 70)
    print("  TIER 1: LINEAR BASELINES (Ridge + ElasticNet)")
    print("=" * 70)

    subset, y = load_and_prepare_data()
    configs = get_feature_configs(subset)
    all_results = []

    for config_name, cfg in configs.items():
        X = subset[cfg["cols"]].values
        print(f"\n{'─'*70}")
        print(f"  {config_name}: {cfg['name']} ({len(cfg['cols'])} features)")
        print(f"{'─'*70}")

        print("\n  Ridge Regression:")
        result = run_cv(build_ridge, X, y, "ridge", config_name)
        all_results.append(result)

        print("\n  ElasticNet:")
        result = run_cv(build_elasticnet, X, y, "elasticnet", config_name)
        all_results.append(result)

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        print(f"  {r['model']:12s} {r['config']}: R²={r['R2_mean']:.4f}±{r['R2_std']:.4f} | "
              f"RMSE={r['RMSE_mean']:.4f}")

    # Significance for each model
    for model_name in ["ridge", "elasticnet"]:
        model_results = [r for r in all_results if r["model"] == model_name]
        print(f"\n  {model_name.upper()}:")
        print_significance(model_results)

    save_results(all_results, "linear_baselines")
    print("=" * 70)


if __name__ == "__main__":
    main()
