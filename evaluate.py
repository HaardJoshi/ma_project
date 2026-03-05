#!/usr/bin/env python3
"""
Evaluation entry point.

Usage:
    python evaluate.py --config configs/financial_only.yaml
    python evaluate.py --config configs/full_fusion.yaml --checkpoint models/fusion_v1/
"""

import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, get_device
from src.evaluation.evaluator import (
    evaluate_sklearn,
    evaluate_pytorch,
    save_results,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate M&A synergy prediction model")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (PyTorch models only)")
    parser.add_argument("--tag", type=str, default="eval",
                        help="Tag for results CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_type = cfg["model"]["type"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    target_col = cfg["preprocessing"]["target_column"]

    print(f"Evaluating model: {model_type}")

    if model_type in ("ridge", "elasticnet", "xgboost"):
        from src.data.dataset import MADealDataset
        from src.models.baselines import build_baseline
        from src.training.trainer import train_sklearn

        train_ds = MADealDataset(processed_dir / "train.csv", target_col=target_col)
        test_ds = MADealDataset(processed_dir / "test.csv", target_col=target_col)

        X_train, y_train = train_ds.features.numpy(), train_ds.labels.numpy()
        X_test, y_test = test_ds.features.numpy(), test_ds.labels.numpy()

        mask_train = ~np.isnan(y_train)
        mask_test = ~np.isnan(y_test)

        model = build_baseline(cfg)
        model.fit(X_train[mask_train], y_train[mask_train])
        metrics = evaluate_sklearn(model, X_test[mask_test], y_test[mask_test])

    elif model_type in ("mlp", "fusion"):
        import torch
        from src.data.dataset import build_dataloaders
        from src.models.mlp import build_mlp

        loaders = build_dataloaders(cfg)
        input_dim = loaders["test"].dataset.features.shape[1]

        if model_type == "mlp":
            model = build_mlp(cfg, input_dim=input_dim)
        else:
            from src.models.fusion import build_fusion
            model = build_fusion(cfg, input_dims={"financial": input_dim})

        if args.checkpoint:
            ckpt_path = Path(args.checkpoint) / "model.pt"
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            print(f"Loaded checkpoint: {ckpt_path}")

        device = get_device(cfg)
        metrics = evaluate_pytorch(model, loaders["test"], device=device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    save_results(metrics, cfg, cfg["output"]["results_dir"], tag=args.tag)
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
