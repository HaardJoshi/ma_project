"""
train_xgboost.py  --  XGBoost Gradient Boosted Trees (Primary Model)
================================================================================
Trains XGBoost regressor across all feature configs (M1, M2, M3).
This is the primary model — expected to outperform linear baselines and MLP
at this sample size (~2,100 deals).

Regularization:
  - max_depth=5, min_child_weight=15: prevent deep/narrow trees
  - subsample=0.8, colsample_bytree=0.8: row/column subsampling (like dropout)
  - reg_alpha=0.1 (L1), reg_lambda=1.0 (L2): weight regularization

Usage:
    python scripts/train_xgboost.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from training_utils import (
    load_and_prepare_data, get_feature_configs, run_cv,
    save_results, print_significance, SEED
)


def build_xgboost():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )),
    ])


def main():
    print("=" * 70)
    print("  TIER 2: XGBOOST (Primary Model)")
    print("=" * 70)

    subset, y = load_and_prepare_data()
    configs = get_feature_configs(subset)
    all_results = []

    for config_name, cfg in configs.items():
        X = subset[cfg["cols"]].values
        print(f"\n{'─'*70}")
        print(f"  {config_name}: {cfg['name']} ({len(cfg['cols'])} features)")
        print(f"{'─'*70}")

        result = run_cv(build_xgboost, X, y, "xgboost", config_name)
        all_results.append(result)

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        print(f"  {r['config']}: R²={r['R2_mean']:.4f}±{r['R2_std']:.4f} | "
              f"RMSE={r['RMSE_mean']:.4f}±{r['RMSE_std']:.4f} | MAE={r['MAE_mean']:.4f}")

    print_significance(all_results)
    save_results(all_results, "xgboost")

    # Feature importance from last fold (informational)
    try:
        last_model = build_xgboost()
        last_model.fit(subset[configs["M3"]["cols"]].values, y)
        importances = last_model.named_steps["model"].feature_importances_
        feat_names = configs["M3"]["cols"]
        top_20 = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:20]
        print(f"\n{'─'*70}")
        print("  TOP 20 FEATURES (M3, full data)")
        print(f"{'─'*70}")
        for name, imp in top_20:
            source = "FIN" if name in configs["M1"]["cols"] else "TXT" if "pca" in name else "GRF"
            print(f"    [{source}] {name:50s} {imp:.4f}")
    except Exception:
        pass

    print("=" * 70)


if __name__ == "__main__":
    main()
