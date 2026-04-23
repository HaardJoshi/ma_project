"""
train_classifier.py  --  Binary Classification: Value-Creating vs Value-Destroying Deals
================================================================================
Reframes CAR prediction as binary classification:
  - Positive class (1): CAR > 0  → "value-creating" deal
  - Negative class (0): CAR ≤ 0  → "value-destroying" deal

Models:
  Tier 1: Logistic Regression (linear baseline)
  Tier 2: XGBoostClassifier (primary)
  Tier 3: MLP with sigmoid (secondary)

Feature configs:
  M1: Financial only (56 features)
  M2: Financial + Text (184 features)
  M3: Financial + Text + Graph (248 features)

Evaluation: 5-fold stratified CV, AUC-ROC, Accuracy, F1, Precision, Recall

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --models xgboost --configs M3
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

from training_utils import (
    load_and_prepare_data, get_feature_configs, save_results,
    SEED, N_FOLDS, TARGET_COL, RESULTS_DIR
)

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── METRICS ─────────────────────────────────────────────────────────────────
def compute_clf_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC_ROC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }


# ── MODEL BUILDERS ──────────────────────────────────────────────────────────
def build_logreg():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0, max_iter=5000, random_state=SEED, class_weight="balanced"
        )),
    ])


def build_xgb_clf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
            scale_pos_weight=1,  # will be set per fold
        )),
    ])


# ── MLP CLASSIFIER ──────────────────────────────────────────────────────────
class ClassifierMLP(nn.Module):
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


def train_mlp_fold(X_train, y_train, X_val, y_val):
    """Train MLP classifier for one fold."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_p = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val_p = scaler.transform(imputer.transform(X_val))

    X_train_t = torch.tensor(X_train_p, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_p, dtype=torch.float32).to(device)

    # Handle class imbalance with weighted loss
    pw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    pos_weight = torch.tensor([pw], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = ClassifierMLP(X_train_p.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, 301):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, torch.tensor(y_val, dtype=torch.float32).to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_val_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

    return preds, probs


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["logreg", "xgboost", "mlp"],
                        choices=["logreg", "xgboost", "mlp"])
    parser.add_argument("--configs", nargs="+",
                        default=["M1", "M2", "M3"],
                        choices=["M1", "M2", "M3"])
    args = parser.parse_args()

    print("=" * 70)
    print("  BINARY CLASSIFICATION: VALUE-CREATING vs VALUE-DESTROYING")
    print("=" * 70)

    subset, y_cont = load_and_prepare_data()
    configs = get_feature_configs(subset)

    # Binary target
    y = (y_cont > 0).astype(int)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"\n  Binary target: {n_pos:,} positive ({100*n_pos/len(y):.1f}%) | "
          f"{n_neg:,} negative ({100*n_neg/len(y):.1f}%)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_results = []

    for config_name in args.configs:
        cfg = configs[config_name]
        X = subset[cfg["cols"]].values

        print(f"\n{'─'*70}")
        print(f"  {config_name}: {cfg['name']} ({len(cfg['cols'])} features)")
        print(f"{'─'*70}")

        for model_name in args.models:
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if model_name == "mlp":
                    y_pred, y_prob = train_mlp_fold(X_train, y_train, X_test, y_test)
                else:
                    if model_name == "logreg":
                        model = build_logreg()
                    elif model_name == "xgboost":
                        model = build_xgb_clf()
                        # Set scale_pos_weight for this fold
                        neg = (y_train == 0).sum()
                        pos = (y_train == 1).sum()
                        model.named_steps["model"].scale_pos_weight = neg / max(pos, 1)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                metrics = compute_clf_metrics(y_test, y_pred, y_prob)
                fold_metrics.append(metrics)

                print(f"    Fold {fold_idx+1}/{N_FOLDS}: "
                      f"Acc={metrics['Accuracy']:.3f} | "
                      f"AUC={metrics['AUC_ROC']:.3f} | "
                      f"F1={metrics['F1']:.3f}")

            # Aggregate
            avg = {m: np.mean([f[m] for f in fold_metrics]) for m in fold_metrics[0]}
            std = {m: np.std([f[m] for f in fold_metrics]) for m in fold_metrics[0]}

            print(f"  ► {model_name:8s}: Acc={avg['Accuracy']:.3f}±{std['Accuracy']:.3f} | "
                  f"AUC={avg['AUC_ROC']:.3f}±{std['AUC_ROC']:.3f} | "
                  f"F1={avg['F1']:.3f}±{std['F1']:.3f}")

            result = {
                "config": config_name,
                "config_name": cfg["name"],
                "model": model_name,
                "n_features": len(cfg["cols"]),
                "Accuracy_mean": avg["Accuracy"], "Accuracy_std": std["Accuracy"],
                "AUC_ROC_mean": avg["AUC_ROC"], "AUC_ROC_std": std["AUC_ROC"],
                "F1_mean": avg["F1"], "F1_std": std["F1"],
                "Precision_mean": avg["Precision"], "Recall_mean": avg["Recall"],
                "fold_aucs": [f["AUC_ROC"] for f in fold_metrics],
            }
            all_results.append(result)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<10s} {'Config':<4s} {'Accuracy':>12s} {'AUC-ROC':>12s} {'F1':>10s}")
    print(f"  {'─'*50}")
    for r in all_results:
        print(f"  {r['model']:<10s} {r['config']:<4s} "
              f"{r['Accuracy_mean']:>6.3f}±{r['Accuracy_std']:.3f} "
              f"{r['AUC_ROC_mean']:>6.3f}±{r['AUC_ROC_std']:.3f} "
              f"{r['F1_mean']:>5.3f}±{r['F1_std']:.3f}")

    # ── Significance tests (M1 vs M3 per model) ─────────────────
    print(f"\n{'─'*70}")
    print("  STATISTICAL SIGNIFICANCE (M1 vs M3, paired t-test on AUC)")
    print(f"{'─'*70}")
    for model_name in args.models:
        m1 = [r for r in all_results if r["config"] == "M1" and r["model"] == model_name]
        m3 = [r for r in all_results if r["config"] == "M3" and r["model"] == model_name]
        if m1 and m3:
            t_stat, p_val = stats.ttest_rel(m3[0]["fold_aucs"], m1[0]["fold_aucs"])
            sig = "✅ p<0.05" if p_val < 0.05 else "❌ p≥0.05"
            delta = m3[0]["AUC_ROC_mean"] - m1[0]["AUC_ROC_mean"]
            print(f"  {model_name:<10s}: AUC Δ={delta:+.4f} | t={t_stat:.3f}, p={p_val:.4f} | {sig}")

    # ── Feature importance (XGBoost M3) ──────────────────────────
    if "xgboost" in args.models and "M3" in args.configs:
        try:
            print(f"\n{'─'*70}")
            print("  TOP 20 FEATURES — XGBoost Classifier (M3)")
            print(f"{'─'*70}")
            m3_cols = configs["M3"]["cols"]
            model = build_xgb_clf()
            model.named_steps["model"].scale_pos_weight = n_neg / max(n_pos, 1)
            model.fit(subset[m3_cols].values, y)
            importances = model.named_steps["model"].feature_importances_
            top = sorted(zip(m3_cols, importances), key=lambda x: x[1], reverse=True)[:20]
            for name, imp in top:
                src = "FIN" if name in configs["M1"]["cols"] else "TXT" if "pca" in name else "GRF"
                print(f"    [{src}] {name:50s} {imp:.4f}")
        except Exception as e:
            print(f"  Feature importance error: {e}")

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{RESULTS_DIR}/classifier_results_{timestamp}.csv"
    import pandas as pd
    df = pd.DataFrame(all_results).drop(columns=["fold_aucs"], errors="ignore")
    df.to_csv(filepath, index=False)
    print(f"\n✅ Results saved to {filepath}")
    print("=" * 70)


if __name__ == "__main__":
    main()
