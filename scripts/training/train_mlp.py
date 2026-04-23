"""
train_mlp.py  --  MLP Fusion Model (Secondary Experiment)
================================================================================
Trains the MLP fusion architecture per methodology §2.1.2:
  Concat [h_fin, h_text, h_struct] → MLP + ReLU + Dropout → CAR

This is a secondary experiment to compare against XGBoost and demonstrate
that tree-based models outperform neural networks at this sample size.

Regularization:
  - Dropout(0.3) after each hidden layer
  - L2 weight decay (1e-4) in Adam optimizer
  - Early stopping (patience=20 epochs)

Usage:
    python scripts/train_mlp.py
    python scripts/train_mlp.py --configs M3   # or M1, M2, M3
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from training_utils import (
    load_and_prepare_data, get_feature_configs, compute_metrics,
    save_results, print_significance, SEED, N_FOLDS
)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── MLP hyperparameters ─────────────────────────────────────────────────────
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
PATIENCE = 20
DROPOUT = 0.3


# ── MODEL ────────────────────────────────────────────────────────────────────
class FusionMLP(nn.Module):
    """
    MLP fusion architecture per §2.1.2.
    Input → 128 → ReLU → Dropout → 64 → ReLU → Dropout → 1
    """
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_fold(X_train, y_train, X_val, y_val):
    """Train MLP on one fold with early stopping."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Preprocessing
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_p = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val_p = scaler.transform(imputer.transform(X_val))

    # Tensors
    X_train_t = torch.tensor(X_train_p, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_p, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = FusionMLP(X_train_p.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    stopped_epoch = EPOCHS

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                stopped_epoch = epoch
                break

    # Restore best
    model.load_state_dict(best_state)

    # Predict
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val_t).cpu().numpy()

    return y_pred, stopped_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["M1", "M2", "M3"])
    args = parser.parse_args()

    print("=" * 70)
    print("  TIER 3: MLP FUSION (Secondary Experiment)")
    print("=" * 70)

    subset, y = load_and_prepare_data()
    configs = get_feature_configs(subset)
    all_results = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for config_name in args.configs:
        cfg = configs[config_name]
        X = subset[cfg["cols"]].values
        print(f"\n{'─'*70}")
        print(f"  {config_name}: {cfg['name']} ({len(cfg['cols'])} features)")
        print(f"{'─'*70}")

        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            y_pred, stopped_epoch = train_one_fold(X_train, y_train, X_test, y_test)
            metrics = compute_metrics(y_test, y_pred)
            fold_metrics.append(metrics)

            print(f"    Fold {fold_idx+1}/{N_FOLDS}: R²={metrics['R2']:.4f}, "
                  f"RMSE={metrics['RMSE']:.4f} (stopped@epoch {stopped_epoch})")

        avg = {m: np.mean([f[m] for f in fold_metrics]) for m in fold_metrics[0]}
        std = {m: np.std([f[m] for f in fold_metrics]) for m in fold_metrics[0]}
        fold_r2s = [f["R2"] for f in fold_metrics]

        print(f"  ► Mean R²={avg['R2']:.4f}±{std['R2']:.4f} | "
              f"RMSE={avg['RMSE']:.4f}±{std['RMSE']:.4f} | MAE={avg['MAE']:.4f}")

        all_results.append({
            "config": config_name,
            "model": "mlp",
            "n_features": X.shape[1],
            "R2_mean": avg["R2"], "R2_std": std["R2"],
            "RMSE_mean": avg["RMSE"], "RMSE_std": std["RMSE"],
            "MAE_mean": avg["MAE"], "MAE_std": std["MAE"],
            "MSE_mean": avg["MSE"], "Huber_mean": avg["Huber"],
            "fold_r2s": fold_r2s,
        })

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        print(f"  {r['config']}: R²={r['R2_mean']:.4f}±{r['R2_std']:.4f} | "
              f"RMSE={r['RMSE_mean']:.4f}±{r['RMSE_std']:.4f}")

    print_significance(all_results)
    save_results(all_results, "mlp")
    print("=" * 70)


if __name__ == "__main__":
    main()
