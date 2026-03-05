#!/usr/bin/env python3
"""
Training entry point.

Usage:
    python train.py --config configs/financial_only.yaml
    python train.py --config configs/full_fusion.yaml --tag experiment_v1
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, get_device
from src.training.trainer import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train M&A synergy prediction model")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment YAML config")
    parser.add_argument("--tag", type=str, default="",
                        help="Experiment tag for checkpoint naming")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["project"]["seed"]
    set_seed(seed)

    model_type = cfg["model"]["type"]
    device = get_device(cfg)
    print(f"Model: {model_type}  |  Device: {device}  |  Seed: {seed}")

    # ── Load data ────────────────────────────────────────────────────
    processed_dir = Path(cfg["data"]["processed_dir"])

    if model_type in ("ridge", "elasticnet", "xgboost"):
        # sklearn path
        from src.data.dataset import MADealDataset
        from src.models.baselines import build_baseline
        from src.training.trainer import train_sklearn
        from src.evaluation.evaluator import save_results

        target_col = cfg["preprocessing"]["target_column"]
        train_ds = MADealDataset(processed_dir / "train.csv", target_col=target_col)
        val_ds = MADealDataset(processed_dir / "val.csv", target_col=target_col)

        import numpy as np
        X_train = train_ds.features.numpy()
        y_train = train_ds.labels.numpy()
        X_val = val_ds.features.numpy()
        y_val = val_ds.labels.numpy()

        # Drop NaN labels
        mask_train = ~np.isnan(y_train)
        mask_val = ~np.isnan(y_val)
        X_train, y_train = X_train[mask_train], y_train[mask_train]
        X_val, y_val = X_val[mask_val], y_val[mask_val]

        model = build_baseline(cfg)
        model, results = train_sklearn(model, X_train, y_train, X_val, y_val)

        print(f"\nTrain MSE: {results['train_mse']:.6f}  R²: {results['train_r2']:.4f}")
        print(f"Val   MSE: {results['val_mse']:.6f}  R²: {results['val_r2']:.4f}")

        save_results(results, cfg, cfg["output"]["results_dir"], tag=args.tag)

    elif model_type in ("mlp", "fusion"):
        # PyTorch path
        from src.data.dataset import build_dataloaders
        from src.models.mlp import build_mlp
        from src.training.trainer import train_pytorch, save_checkpoint

        loaders = build_dataloaders(cfg)
        input_dim = loaders["train"].dataset.features.shape[1]

        if model_type == "mlp":
            model = build_mlp(cfg, input_dim=input_dim)
        else:
            from src.models.fusion import build_fusion
            model = build_fusion(cfg, input_dims={"financial": input_dim})

        print(f"\nModel architecture:\n{model}\n")

        model, history = train_pytorch(model, loaders["train"], loaders["val"], cfg)
        save_checkpoint(model, cfg, history, cfg["output"]["model_dir"], tag=args.tag)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
