"""
test_h1.py  --  H1: Topological Alpha — Sector-Segmented Analysis
================================================================================
Tests if graph embeddings improve prediction MORE in supply-chain-dependent
sectors (Manufacturing/Transport SIC 20-49) than asset-light sectors
(Finance/Services SIC 60-79).

Usage:
    python scripts/test_h1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from training_utils import (
    load_and_prepare_data, get_feature_configs, SEED, N_FOLDS
)

np.random.seed(SEED)


def run_cv(X, y, n_folds):
    """Run stratified CV with XGBoost and return fold AUCs."""
    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = neg / max(pos, 1)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_aucs = []

    for train_idx, test_idx in skf.split(X, y):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                objective="binary:logistic", scale_pos_weight=spw,
                random_state=SEED, n_jobs=-1, verbosity=0,
            )),
        ])
        model.fit(X[train_idx], y[train_idx])
        y_prob = model.predict_proba(X[test_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y[test_idx], y_prob))

    return fold_aucs


def main():
    print("=" * 70)
    print("  H1: TOPOLOGICAL ALPHA — SECTOR-SEGMENTED ANALYSIS")
    print("=" * 70)

    subset, y_cont = load_and_prepare_data()
    configs = get_feature_configs(subset)
    y = (y_cont > 0).astype(int)

    sic_col = "Current Acquirer SIC Code"
    if sic_col not in subset.columns:
        print("  ❌ No SIC codes available")
        return

    sic_2digit = subset[sic_col].astype(str).str[:2]

    # Define sector groups
    sc_dependent = [str(i) for i in range(20, 50)]   # Manufacturing + Transport
    asset_light = [str(i) for i in range(60, 68)] + ["70", "73", "78", "79"]  # Finance + Services

    sector_groups = {
        "supply_chain": sc_dependent,
        "asset_light": asset_light,
    }

    results = {}

    for group_name, sic_list in sector_groups.items():
        mask = sic_2digit.isin(sic_list)
        n_deals = mask.sum()
        y_group = y[mask.values]
        n_pos = (y_group == 1).sum()

        print(f"\n{'─'*70}")
        print(f"  {group_name.upper()} ({n_deals} deals, {n_pos} positive / {n_deals-n_pos} negative)")
        print(f"{'─'*70}")

        if n_deals < 50:
            print(f"    ⚠️ Too few deals — skipping")
            continue

        n_folds = min(N_FOLDS, max(2, n_deals // 30))

        for config_name in ["M1", "M3"]:
            X = subset.loc[mask, configs[config_name]["cols"]].values
            fold_aucs = run_cv(X, y_group, n_folds)
            avg = np.mean(fold_aucs)
            std = np.std(fold_aucs)
            print(f"    {config_name} ({len(configs[config_name]['cols'])} feat): "
                  f"AUC={avg:.3f}±{std:.3f}")
            results[(group_name, config_name)] = {"auc": avg, "folds": fold_aucs}

    # Compare
    print(f"\n{'='*70}")
    print("  SECTOR COMPARISON")
    print(f"{'='*70}")

    deltas = {}
    for group in ["supply_chain", "asset_light"]:
        if (group, "M1") in results and (group, "M3") in results:
            m1 = results[(group, "M1")]
            m3 = results[(group, "M3")]
            delta = m3["auc"] - m1["auc"]
            deltas[group] = delta

            t_stat, p_val = stats.ttest_rel(m3["folds"], m1["folds"])
            sig = "✅ p<0.05" if p_val < 0.05 else "❌ p≥0.05"
            print(f"  {group:15s}: M1={m1['auc']:.3f} → M3={m3['auc']:.3f} | "
                  f"Δ={delta:+.4f} | t={t_stat:.3f}, p={p_val:.4f} {sig}")

    if deltas:
        sc_d = deltas.get("supply_chain", 0)
        al_d = deltas.get("asset_light", 0)
        print(f"\n  Supply-chain Δ: {sc_d:+.4f}")
        print(f"  Asset-light Δ:  {al_d:+.4f}")

        if sc_d > al_d:
            print(f"\n  ✅ H1 SUPPORTED: Graph helps more in supply-chain sectors")
        else:
            print(f"\n  ❌ H1 NOT SUPPORTED: Graph helps more in asset-light sectors")

    print("=" * 70)


if __name__ == "__main__":
    main()
