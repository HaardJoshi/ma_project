"""
Training loop, cross-validation, and checkpointing.

Supports:
- sklearn-style baselines (Ridge, XGBoost) via .fit()/.predict()
- PyTorch models (MLP, Fusion) via custom training loop
- k-fold cross-validation
- Early stopping with patience
- Checkpoint saving / loading
"""

import os
import json
import math
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import get_device


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────
# Sklearn-style training (Ridge, XGBoost)
# ─────────────────────────────────────────────────────────────────────

def train_sklearn(model, X_train, y_train, X_val, y_val):
    """
    Train an sklearn-compatible model (Ridge, XGBoost, etc.).

    Returns
    -------
    dict
        Training results with train/val metrics.
    """
    from sklearn.metrics import mean_squared_error, r2_score

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    results = {
        "train_mse": float(mean_squared_error(y_train, y_pred_train)),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "val_mse": float(mean_squared_error(y_val, y_pred_val)),
        "val_r2": float(r2_score(y_val, y_pred_val)),
    }
    return model, results


# ─────────────────────────────────────────────────────────────────────
# PyTorch training loop (MLP, Fusion)
# ─────────────────────────────────────────────────────────────────────

def train_pytorch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
) -> tuple[nn.Module, dict]:
    """
    Train a PyTorch model with early stopping and checkpointing.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    cfg : dict
        Full config dictionary.

    Returns
    -------
    tuple[nn.Module, dict]
        Best model and training history.
    """
    device = get_device(cfg)
    model = model.to(device)

    lr = cfg["training"]["learning_rate"]
    wd = cfg["training"]["weight_decay"]
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 3,
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Skip NaN labels
            mask = ~torch.isnan(y_batch)
            if mask.sum() == 0:
                continue

            pred = model(X_batch[mask]).squeeze(-1)
            loss = criterion(pred, y_batch[mask])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                mask = ~torch.isnan(y_batch)
                if mask.sum() == 0:
                    continue
                pred = model(X_batch[mask]).squeeze(-1)
                loss = criterion(pred, y_batch[mask])
                val_losses.append(loss.item())

        avg_train = sum(train_losses) / max(len(train_losses), 1)
        avg_val = sum(val_losses) / max(len(val_losses), 1)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        scheduler.step(avg_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{epochs}  "
                  f"train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")

        # ── Early stopping ───────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def save_checkpoint(
    model: nn.Module,
    cfg: dict,
    history: dict,
    output_dir: str | Path,
    tag: str = "",
) -> str:
    """Save model checkpoint + config + history to disk."""
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{cfg['model']['type']}_{tag}_{timestamp}" if tag else f"{cfg['model']['type']}_{timestamp}"
    ckpt_dir = output_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / "model.pt")

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"✅ Checkpoint saved → {ckpt_dir}")
    return str(ckpt_dir)
