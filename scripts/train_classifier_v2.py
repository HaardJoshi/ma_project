"""
train_classifier_v2.py  --  Enhanced Classifier with Feature Engineering + SHAP
================================================================================
Adds engineered interaction features to the classifier pipeline:
  - Acquirer/Target financial ratios (size ratio, profitability gap, etc.)
  - Cross-modal interactions (financial × text/graph signals)
  - SHAP analysis for the best model

Usage:
    python scripts/train_classifier_v2.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier

from training_utils import (
    load_and_prepare_data, get_feature_configs,
    SEED, N_FOLDS, RESULTS_DIR
)

np.random.seed(SEED)


# ── FEATURE ENGINEERING ─────────────────────────────────────────────────────
def engineer_features(df):
    """Create interaction and ratio features."""
    fe = pd.DataFrame(index=df.index)

    # Size ratios
    fe["size_ratio"] = df.get("Target Total Assets", 0) / df.get("Acquirer Total Assets", 1).replace(0, np.nan)
    fe["mcap_ratio"] = df.get("Target Market Value of Equity", 0) / df.get("Acquirer Current Market Cap", 1).replace(0, np.nan)
    fe["revenue_ratio"] = df.get("Target Sales/Revenue/Turnover", 0) / df.get("Acquirer Sales/Revenue/Turnover", 1).replace(0, np.nan)

    # Profitability gaps
    fe["margin_gap"] = df.get("Acquirer Operating Margin", 0) - df.get("Target Operating Margin", 0)
    fe["roe_gap"] = df.get("Acquirer Return on Common Equity", 0) - df.get("Target Return on Common Equity", 0)
    fe["ebitda_gap"] = df.get("Acquirer EBITDA(Earn Bef Int Dep & Amo)", 0) - df.get("Target EBITDA(Earn Bef Int Dep & Amo)", 0)

    # Leverage gaps
    fe["leverage_gap"] = df.get("Acquirer Total Debt to Total Assets", 0) - df.get("Target Total Debt to Total Assets", 0)
    fe["liquidity_gap"] = df.get("Acquirer Current Ratio", 0) - df.get("Target Current Ratio", 0)

    # Growth gaps
    fe["rev_growth_gap"] = df.get("Acquirer Net Revenue Growth", 0) - df.get("Target Net Revenue Growth", 0)
    fe["asset_growth_gap"] = df.get("Acquirer Asset Growth", 0) - df.get("Target Asset Growth", 0)

    # Deal characteristics
    fe["relative_deal_size"] = df.get("Announced Total Value (mil.)", 0) / df.get("Acquirer Current Market Cap", 1).replace(0, np.nan)
    fe["cash_heavy"] = (df.get("Payment_Cash", 0) > 0.5).astype(float)
    fe["stock_heavy"] = (df.get("Payment_Stock", 0) > 0.5).astype(float)

    # Replace infinities and clip extreme ratios
    fe = fe.replace([np.inf, -np.inf], np.nan)
    for col in fe.columns:
        if fe[col].notna().sum() > 0:
            lo, hi = fe[col].quantile(0.01), fe[col].quantile(0.99)
            fe[col] = fe[col].clip(lo, hi)

    return fe


def run_experiment(X, y, feature_names, config_name, config_desc):
    """Run 5-fold CV with XGBoost classifier."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / max(pos, 1)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                objective="binary:logistic", eval_metric="auc",
                scale_pos_weight=spw, random_state=SEED,
                n_jobs=-1, verbosity=0,
            )),
        ])

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        metrics = {
            "AUC_ROC": roc_auc_score(y_test, y_prob),
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
        }
        fold_metrics.append(metrics)
        print(f"    Fold {fold_idx+1}/{N_FOLDS}: AUC={metrics['AUC_ROC']:.3f} | "
              f"Acc={metrics['Accuracy']:.3f} | F1={metrics['F1']:.3f}")

    avg = {m: np.mean([f[m] for f in fold_metrics]) for m in fold_metrics[0]}
    std = {m: np.std([f[m] for f in fold_metrics]) for m in fold_metrics[0]}
    fold_aucs = [f["AUC_ROC"] for f in fold_metrics]

    print(f"  ► AUC={avg['AUC_ROC']:.3f}±{std['AUC_ROC']:.3f} | "
          f"Acc={avg['Accuracy']:.3f} | F1={avg['F1']:.3f}")

    return {
        "config": config_name,
        "config_desc": config_desc,
        "n_features": X.shape[1],
        "AUC_mean": avg["AUC_ROC"], "AUC_std": std["AUC_ROC"],
        "Acc_mean": avg["Accuracy"], "F1_mean": avg["F1"],
        "fold_aucs": fold_aucs,
    }, model


def run_shap_analysis(model, X, feature_names, config_name):
    """Generate SHAP analysis for the model."""
    try:
        import shap

        print(f"\n{'─'*70}")
        print(f"  SHAP ANALYSIS — {config_name}")
        print(f"{'─'*70}")

        # Preprocess X through the pipeline's imputer
        X_imputed = model.named_steps["imputer"].transform(X)
        explainer = shap.TreeExplainer(model.named_steps["model"])
        shap_values = explainer.shap_values(X_imputed)

        # Mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_shap)[::-1][:20]

        print(f"\n  Top 20 features by mean |SHAP|:")
        for rank, idx in enumerate(top_indices, 1):
            name = feature_names[idx]
            if "pca" in name:
                src = "TXT"
            elif "graph" in name:
                src = "GRF"
            elif name in ["size_ratio", "mcap_ratio", "revenue_ratio", "margin_gap",
                          "roe_gap", "ebitda_gap", "leverage_gap", "liquidity_gap",
                          "rev_growth_gap", "asset_growth_gap", "relative_deal_size",
                          "cash_heavy", "stock_heavy"]:
                src = "ENG"
            else:
                src = "FIN"
            print(f"    {rank:2d}. [{src}] {name:45s} {mean_shap[idx]:.4f}")

        # Save SHAP values
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        shap_df.to_csv(f"{RESULTS_DIR}/shap_values_{config_name}.csv", index=False)
        print(f"\n  Saved SHAP values to results/shap_values_{config_name}.csv")

    except Exception as e:
        print(f"  SHAP error: {e}")


def main():
    print("=" * 70)
    print("  ENHANCED CLASSIFIER WITH FEATURE ENGINEERING + SHAP")
    print("=" * 70)

    subset, y_cont = load_and_prepare_data()
    configs = get_feature_configs(subset)
    y = (y_cont > 0).astype(int)

    print(f"\n  Binary target: {y.sum():,} positive ({100*y.mean():.1f}%) | "
          f"{len(y)-y.sum():,} negative")

    # Engineer features
    print("\n  Engineering interaction features...")
    fe = engineer_features(subset)
    eng_cols = list(fe.columns)
    print(f"  Created {len(eng_cols)} engineered features: {eng_cols}")

    # Combine
    subset_enhanced = pd.concat([subset, fe], axis=1)

    # Define enhanced configs
    configs_enhanced = {
        "M1": {"name": "Financial Only", "cols": configs["M1"]["cols"]},
        "M1e": {"name": "Financial + Engineered", "cols": configs["M1"]["cols"] + eng_cols},
        "M3": {"name": "Fin + Text + Graph", "cols": configs["M3"]["cols"]},
        "M3e": {"name": "Fin + Text + Graph + Engineered", "cols": configs["M3"]["cols"] + eng_cols},
    }

    all_results = []
    best_model = None
    best_auc = 0
    best_config = None
    best_features = None

    for config_name, cfg in configs_enhanced.items():
        X = subset_enhanced[cfg["cols"]].values
        print(f"\n{'─'*70}")
        print(f"  {config_name}: {cfg['name']} ({len(cfg['cols'])} features)")
        print(f"{'─'*70}")

        result, model = run_experiment(X, y, cfg["cols"], config_name, cfg["name"])
        all_results.append(result)

        if result["AUC_mean"] > best_auc:
            best_auc = result["AUC_mean"]
            best_model = model
            best_config = config_name
            best_features = cfg["cols"]

    # Summary
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")

    # Baselines from untuned run
    baselines = {"M1": 0.541, "M3": 0.566}

    print(f"  {'Config':<8s} {'Features':>8s} {'AUC':>14s} {'vs Base':>8s} {'Acc':>6s} {'F1':>6s}")
    print(f"  {'─'*52}")
    for r in all_results:
        base_key = r["config"].rstrip("e")
        baseline = baselines.get(base_key, 0.5)
        delta = r["AUC_mean"] - baseline
        print(f"  {r['config']:<8s} {r['n_features']:>8d} "
              f"{r['AUC_mean']:>6.3f}±{r['AUC_std']:.3f} "
              f"{delta:>+8.3f} {r['Acc_mean']:>6.3f} {r['F1_mean']:>6.3f}")

    # Significance tests
    print(f"\n{'─'*70}")
    print("  SIGNIFICANCE TESTS (paired t-test on AUC)")
    print(f"{'─'*70}")

    pairs = [("M1", "M3"), ("M1", "M1e"), ("M3", "M3e"), ("M1", "M3e")]
    for a, b in pairs:
        ra = [r for r in all_results if r["config"] == a]
        rb = [r for r in all_results if r["config"] == b]
        if ra and rb:
            t_stat, p_val = stats.ttest_rel(rb[0]["fold_aucs"], ra[0]["fold_aucs"])
            sig = "✅ p<0.05" if p_val < 0.05 else "❌ p≥0.05"
            delta = rb[0]["AUC_mean"] - ra[0]["AUC_mean"]
            print(f"  {a} vs {b}: Δ={delta:+.4f} | t={t_stat:.3f}, p={p_val:.4f} | {sig}")

    # SHAP for best model
    if best_model and best_features:
        X_full = subset_enhanced[best_features].values
        run_shap_analysis(best_model, X_full, best_features, best_config)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{RESULTS_DIR}/enhanced_classifier_results_{timestamp}.csv"
    save_df = pd.DataFrame([{k: v for k, v in r.items() if k != "fold_aucs"} for r in all_results])
    save_df.to_csv(filepath, index=False)
    print(f"\n✅ Results saved to {filepath}")
    print("=" * 70)


if __name__ == "__main__":
    main()
