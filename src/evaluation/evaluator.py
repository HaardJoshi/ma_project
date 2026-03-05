"""
Model evaluation — metrics computation and results export.

Computes MSE, R², MAE on the test set and exports results
as a CSV / JSON for comparison across experiments.
"""

import csv
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground truth and predicted values.

    Returns
    -------
    dict
        MSE, RMSE, MAE, R².
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    n = len(y_true)
    if n == 0:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "n": n}


def evaluate_sklearn(model, X_test, y_test) -> dict:
    """Evaluate an sklearn model on the test set."""
    y_pred = model.predict(X_test)
    return compute_metrics(np.array(y_test), np.array(y_pred))


def evaluate_pytorch(model, test_loader, device: str = "cpu") -> dict:
    """
    Evaluate a PyTorch model on the test DataLoader.

    Returns
    -------
    dict
        Regression metrics.
    """
    import torch

    model.eval()
    model = model.to(device)
    all_true, all_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            mask = ~torch.isnan(y_batch)
            if mask.sum() == 0:
                continue
            pred = model(X_batch[mask]).squeeze(-1).cpu().numpy()
            true = y_batch[mask].numpy()
            all_true.extend(true.tolist())
            all_pred.extend(pred.tolist())

    return compute_metrics(np.array(all_true), np.array(all_pred))


def save_results(
    metrics: dict,
    cfg: dict,
    results_dir: str | Path,
    tag: str = "",
) -> str:
    """
    Append evaluation results to the results CSV.

    Parameters
    ----------
    metrics : dict
        Output of compute_metrics.
    cfg : dict
        Experiment configuration.
    results_dir : str | Path
        Directory to save results in.
    tag : str
        Optional experiment tag.

    Returns
    -------
    str
        Path to the results CSV.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "experiment_results.csv"

    row = {
        "timestamp": datetime.now().isoformat(),
        "model_type": cfg.get("model", {}).get("type", ""),
        "tag": tag,
        "features_financial": cfg.get("features", {}).get("financial", False),
        "features_text": cfg.get("features", {}).get("text", False),
        "features_graph": cfg.get("features", {}).get("graph", False),
        **metrics,
    }

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"✅ Results appended → {csv_path}")
    print(f"   MSE={metrics['mse']:.6f}  R²={metrics['r2']:.4f}  n={metrics['n']}")
    return str(csv_path)
