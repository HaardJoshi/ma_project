"""
tune_xgboost.py  --  Bayesian Hyperparameter Tuning with Optuna
================================================================================
Uses Optuna to optimise XGBoostClassifier hyperparameters for M1, M2, and M3
feature configs. Inner 3-fold CV for objective, outer 5-fold for final eval.

Usage:
    python scripts/tune_xgboost.py                     # tune all configs
    python scripts/tune_xgboost.py --configs M3         # tune M3 only
    python scripts/tune_xgboost.py --n_trials 200       # more trials
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import optuna
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier

from training_utils import (
    load_and_prepare_data, get_feature_configs,
    SEED, N_FOLDS, RESULTS_DIR
)

np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_objective(X, y, n_inner_folds=3):
    """Create Optuna objective for XGBoost classifier."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / max(pos, 1)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="auc",
                scale_pos_weight=spw,
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
            )),
        ])

        skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=SEED)
        scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    return objective


def evaluate_best(best_params, X, y, config_name):
    """Evaluate best params with outer 5-fold CV."""
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
                **best_params,
                objective="binary:logistic",
                eval_metric="auc",
                scale_pos_weight=spw,
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
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

    print(f"  ► TUNED: AUC={avg['AUC_ROC']:.3f}±{std['AUC_ROC']:.3f} | "
          f"Acc={avg['Accuracy']:.3f}±{std['Accuracy']:.3f} | "
          f"F1={avg['F1']:.3f}±{std['F1']:.3f}")

    return avg, std, fold_aucs, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["M1", "M2", "M3"])
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()

    print("=" * 70)
    print("  OPTUNA HYPERPARAMETER TUNING — XGBoost Classifier")
    print(f"  Trials per config: {args.n_trials}")
    print("=" * 70)

    subset, y_cont = load_and_prepare_data()
    configs = get_feature_configs(subset)
    y = (y_cont > 0).astype(int)

    print(f"\n  Binary target: {y.sum():,} positive ({100*y.mean():.1f}%) | "
          f"{len(y)-y.sum():,} negative")

    # Baseline (untuned) results for comparison
    baseline_aucs = {"M1": 0.541, "M2": 0.529, "M3": 0.566}
    all_results = []

    for config_name in args.configs:
        cfg = configs[config_name]
        X = subset[cfg["cols"]].values

        print(f"\n{'─'*70}")
        print(f"  {config_name}: {cfg['name']} ({len(cfg['cols'])} features)")
        print(f"{'─'*70}")

        # Tune
        print(f"  Running {args.n_trials} Optuna trials...")
        objective = create_objective(X, y)
        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_inner_auc = study.best_value

        print(f"\n  Best inner CV AUC: {best_inner_auc:.4f}")
        print(f"  Best params:")
        for k, v in best_params.items():
            print(f"    {k}: {v}")

        # Evaluate with outer CV
        print(f"\n  Outer 5-fold evaluation:")
        avg, std, fold_aucs, model = evaluate_best(best_params, X, y, config_name)

        baseline = baseline_aucs.get(config_name, 0.5)
        delta = avg["AUC_ROC"] - baseline
        print(f"\n  vs baseline: {baseline:.3f} → {avg['AUC_ROC']:.3f} (Δ={delta:+.3f})")

        all_results.append({
            "config": config_name,
            "config_name": cfg["name"],
            "n_features": len(cfg["cols"]),
            "AUC_baseline": baseline,
            "AUC_tuned_mean": avg["AUC_ROC"],
            "AUC_tuned_std": std["AUC_ROC"],
            "AUC_delta": delta,
            "Acc_mean": avg["Accuracy"],
            "F1_mean": avg["F1"],
            "fold_aucs": fold_aucs,
            "best_params": best_params,
        })

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TUNING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<6s} {'Baseline':>10s} {'Tuned':>14s} {'Δ AUC':>8s}")
    print(f"  {'─'*40}")
    for r in all_results:
        print(f"  {r['config']:<6s} {r['AUC_baseline']:>10.3f} "
              f"{r['AUC_tuned_mean']:>6.3f}±{r['AUC_tuned_std']:.3f} "
              f"{r['AUC_delta']:>+8.3f}")

    # Significance M1 vs M3 (tuned)
    m1 = [r for r in all_results if r["config"] == "M1"]
    m3 = [r for r in all_results if r["config"] == "M3"]
    if m1 and m3:
        t_stat, p_val = stats.ttest_rel(m3[0]["fold_aucs"], m1[0]["fold_aucs"])
        sig = "✅ p<0.05" if p_val < 0.05 else "❌ p≥0.05"
        delta = m3[0]["AUC_tuned_mean"] - m1[0]["AUC_tuned_mean"]
        print(f"\n  M1 vs M3 (tuned): AUC Δ={delta:+.4f} | t={t_stat:.3f}, p={p_val:.4f} | {sig}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{RESULTS_DIR}/tuned_xgboost_results_{timestamp}.csv"
    save_data = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k not in ("fold_aucs", "best_params")}
        row.update({f"param_{k}": v for k, v in r["best_params"].items()})
        save_data.append(row)
    pd.DataFrame(save_data).to_csv(filepath, index=False)
    print(f"\n✅ Results saved to {filepath}")
    print("=" * 70)


if __name__ == "__main__":
    main()
