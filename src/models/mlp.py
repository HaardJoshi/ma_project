"""
Shallow MLP predictor.

A 1–2 hidden-layer MLP with dropout and L2 regularisation,
used as the prediction head for single-stream (Block A) or
multi-stream (fusion) experiments.
"""

import torch
import torch.nn as nn


class SynergyMLP(nn.Module):
    """
    Multi-layer perceptron for CAR prediction.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list[int]
        Sizes of hidden layers (e.g. [128, 64]).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64],
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))  # single output = predicted CAR
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → (batch_size, 1)."""
        return self.net(x)


def build_mlp(cfg: dict, input_dim: int) -> SynergyMLP:
    """
    Build MLP from config.

    Parameters
    ----------
    cfg : dict
        Full config (reads ``cfg["model"]["mlp"]``).
    input_dim : int
        Number of input features.

    Returns
    -------
    SynergyMLP
    """
    mlp_cfg = cfg.get("model", {}).get("mlp", {})
    return SynergyMLP(
        input_dim=input_dim,
        hidden_dims=mlp_cfg.get("hidden_dims", [128, 64]),
        dropout=mlp_cfg.get("dropout", 0.3),
    )
