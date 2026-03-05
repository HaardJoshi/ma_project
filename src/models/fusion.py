"""
Dual-stream fusion model.

Concatenates feature embeddings from multiple blocks:
    z_i = [ h_F ∥ h_T ∥ h_G ]
and passes through a prediction head to output predicted CAR.

Each stream has its own projection head that maps raw features
to a lower-dimensional embedding before concatenation.
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Linear projection + ReLU for one feature stream."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionModel(nn.Module):
    """
    Multi-stream fusion model.

    Parameters
    ----------
    financial_input_dim : int
        Raw dimension of h_F.
    financial_proj_dim : int
        Projected dimension for financial stream.
    text_input_dim : int
        Raw dimension of h_T (0 if text disabled).
    text_proj_dim : int
        Projected dimension for text stream.
    graph_input_dim : int
        Raw dimension of h_G (0 if graph disabled).
    graph_proj_dim : int
        Projected dimension for graph stream.
    head_hidden : int
        Hidden dim of prediction head.
    head_dropout : float
        Dropout in prediction head.
    """

    def __init__(
        self,
        financial_input_dim: int,
        financial_proj_dim: int = 64,
        text_input_dim: int = 0,
        text_proj_dim: int = 64,
        graph_input_dim: int = 0,
        graph_proj_dim: int = 32,
        head_hidden: int = 64,
        head_dropout: float = 0.3,
    ):
        super().__init__()

        self.has_text = text_input_dim > 0
        self.has_graph = graph_input_dim > 0

        # Stream projections
        self.financial_proj = ProjectionHead(financial_input_dim, financial_proj_dim)
        self.text_proj = ProjectionHead(text_input_dim, text_proj_dim) if self.has_text else None
        self.graph_proj = ProjectionHead(graph_input_dim, graph_proj_dim) if self.has_graph else None

        # Fusion dimension
        fused_dim = financial_proj_dim
        if self.has_text:
            fused_dim += text_proj_dim
        if self.has_graph:
            fused_dim += graph_proj_dim

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        h_f: torch.Tensor,
        h_t: torch.Tensor | None = None,
        h_g: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        h_f : Tensor (batch, financial_input_dim)
        h_t : Tensor (batch, text_input_dim), optional
        h_g : Tensor (batch, graph_input_dim), optional

        Returns
        -------
        Tensor (batch, 1) — predicted CAR
        """
        parts = [self.financial_proj(h_f)]
        if self.has_text and h_t is not None:
            parts.append(self.text_proj(h_t))
        if self.has_graph and h_g is not None:
            parts.append(self.graph_proj(h_g))

        z = torch.cat(parts, dim=1)
        return self.head(z)


def build_fusion(cfg: dict, input_dims: dict[str, int]) -> FusionModel:
    """
    Build the fusion model from config.

    Parameters
    ----------
    cfg : dict
        Full config (reads ``cfg["model"]["fusion"]``).
    input_dims : dict
        Mapping of stream name → raw input dimension.
        Required key: "financial". Optional: "text", "graph".

    Returns
    -------
    FusionModel
    """
    fc = cfg.get("model", {}).get("fusion", {})
    return FusionModel(
        financial_input_dim=input_dims["financial"],
        financial_proj_dim=fc.get("financial_dim", 64),
        text_input_dim=input_dims.get("text", 0),
        text_proj_dim=fc.get("text_dim", 64),
        graph_input_dim=input_dims.get("graph", 0),
        graph_proj_dim=fc.get("graph_dim", 32),
        head_hidden=fc.get("head_hidden", 64),
        head_dropout=fc.get("head_dropout", 0.3),
    )
