"""
train_hetero_graph.py  --  Train Heterogeneous GraphSAGE & extract embeddings
================================================================================
Trains a 2-layer heterogeneous GraphSAGE model using self-supervised link
prediction. Uses HeteroConv with separate SAGEConv per edge type:
  - "supplies" edges: supplier → acquirer
  - "buys_from" edges: acquirer → customer

Each edge type learns its own aggregation function, matching the methodology
requirement for heterogeneous message-passing (§2.1.2).

Output: data/interim/hetero_graph_embeddings.csv
"""

import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

# ── CONFIG ──────────────────────────────────────────────────────────────────
GRAPH_FILE  = "data/interim/hetero_supply_chain_graph.pt"
META_FILE   = "data/interim/hetero_graph_metadata.json"
OUTPUT_CSV  = "data/interim/hetero_graph_embeddings.csv"

HIDDEN_DIM  = 128
EMBED_DIM   = 64
EPOCHS      = 200
LR          = 0.01
TEST_RATIO  = 0.15
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── MODEL ────────────────────────────────────────────────────────────────────
class HeteroGraphSAGE(torch.nn.Module):
    """
    2-layer Heterogeneous GraphSAGE.

    Each layer uses HeteroConv wrapping separate SAGEConv per edge type.
    This means "supplies" edges and "buys_from" edges learn independent
    aggregation functions — the key difference from the homogeneous version.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # Layer 1: separate SAGEConv per edge type
        self.conv1 = HeteroConv({
            ("company", "supplies", "company"): SAGEConv(in_channels, hidden_channels),
            ("company", "buys_from", "company"): SAGEConv(in_channels, hidden_channels),
        }, aggr="mean")  # aggregate across edge types via mean

        # Layer 2: separate SAGEConv per edge type
        self.conv2 = HeteroConv({
            ("company", "supplies", "company"): SAGEConv(hidden_channels, out_channels),
            ("company", "buys_from", "company"): SAGEConv(hidden_channels, out_channels),
        }, aggr="mean")

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=0.3, training=self.training) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

    def encode(self, x_dict, edge_index_dict):
        """Get node embeddings (no dropout)."""
        self.eval()
        with torch.no_grad():
            x_dict = self.conv1(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


def decode(z, edge_index):
    """Predict edge likelihood via dot product."""
    src, dst = edge_index
    return (z[src] * z[dst]).sum(dim=1)


def train_epoch(model, optimizer, data, train_edges_dict):
    model.train()
    optimizer.zero_grad()

    x_dict = {"company": data["company"].x}
    z_dict = model(x_dict, train_edges_dict)
    z = z_dict["company"]

    total_loss = 0.0
    n_edge_types = 0

    for edge_type_key, edge_index in train_edges_dict.items():
        # Positive edges
        pos_pred = decode(z, edge_index)

        # Negative edges
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=data["company"].num_nodes,
            num_neg_samples=edge_index.size(1),
        )
        neg_pred = decode(z, neg_edge_index)

        pos_labels = torch.ones(pos_pred.size(0))
        neg_labels = torch.zeros(neg_pred.size(0))

        preds = torch.cat([pos_pred, neg_pred])
        labels = torch.cat([pos_labels, neg_labels])

        loss = F.binary_cross_entropy_with_logits(preds, labels)
        total_loss += loss
        n_edge_types += 1

    # Average loss across edge types
    total_loss = total_loss / n_edge_types
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


@torch.no_grad()
def evaluate(model, data, train_edges_dict, test_edges_dict):
    model.eval()
    x_dict = {"company": data["company"].x}
    z_dict = model(x_dict, train_edges_dict)
    z = z_dict["company"]

    all_labels = []
    all_preds = []

    for edge_type_key, test_edge_index in test_edges_dict.items():
        train_edge_index = train_edges_dict[edge_type_key]

        # Positive test
        pos_pred = decode(z, test_edge_index).sigmoid().cpu().numpy()

        # Negative test
        neg_edge_index = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=data["company"].num_nodes,
            num_neg_samples=test_edge_index.size(1),
        )
        neg_pred = decode(z, neg_edge_index).sigmoid().cpu().numpy()

        all_labels.extend([1] * len(pos_pred))
        all_labels.extend([0] * len(neg_pred))
        all_preds.extend(pos_pred.tolist())
        all_preds.extend(neg_pred.tolist())

    auc = roc_auc_score(all_labels, all_preds)
    return auc


def split_edges(edge_index, test_ratio):
    """Split edges into train/test sets."""
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    n_test = int(num_edges * test_ratio)
    return edge_index[:, perm[n_test:]], edge_index[:, perm[:n_test]]


def main():
    print("=" * 60)
    print("  TRAINING HETEROGENEOUS GRAPHSAGE (LINK PREDICTION)")
    print("=" * 60)

    # Load graph and metadata
    data = torch.load(GRAPH_FILE, weights_only=False)
    with open(META_FILE, "r") as f:
        meta = json.load(f)

    print(f"Graph: {data['company'].num_nodes:,} nodes")
    supplies_edges = data["company", "supplies", "company"].edge_index
    buys_edges = data["company", "buys_from", "company"].edge_index
    print(f"  'supplies' edges:  {supplies_edges.size(1):,}")
    print(f"  'buys_from' edges: {buys_edges.size(1):,}")

    # ── Train/test edge split (per edge type) ────────────────────
    supply_train, supply_test = split_edges(supplies_edges, TEST_RATIO)
    buys_train, buys_test = split_edges(buys_edges, TEST_RATIO)

    train_edges_dict = {
        ("company", "supplies", "company"): supply_train,
        ("company", "buys_from", "company"): buys_train,
    }
    test_edges_dict = {
        ("company", "supplies", "company"): supply_test,
        ("company", "buys_from", "company"): buys_test,
    }

    print(f"\nTrain edges: supplies={supply_train.size(1):,}, buys_from={buys_train.size(1):,}")
    print(f"Test edges:  supplies={supply_test.size(1):,}, buys_from={buys_test.size(1):,}")

    # ── Initialize model ─────────────────────────────────────────
    in_channels = data["company"].x.shape[1]
    model = HeteroGraphSAGE(in_channels, HIDDEN_DIM, EMBED_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nModel: HeteroGraphSAGE({in_channels} → {HIDDEN_DIM} → {EMBED_DIM})")
    print(f"  Separate SAGEConv for 'supplies' and 'buys_from' edges")
    print(f"Training for {EPOCHS} epochs...")

    # ── Training loop ────────────────────────────────────────────
    best_auc = 0
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, optimizer, data, train_edges_dict)

        if epoch % 20 == 0 or epoch == 1:
            auc = evaluate(model, data, train_edges_dict, test_edges_dict)
            if auc > best_auc:
                best_auc = auc
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Test AUC: {auc:.4f}")

    print(f"\nBest test AUC: {best_auc:.4f}")

    # ── Extract embeddings ───────────────────────────────────────
    print("\nExtracting node embeddings...")

    # Use full edges for final embedding extraction
    full_edges_dict = {
        ("company", "supplies", "company"): supplies_edges,
        ("company", "buys_from", "company"): buys_edges,
    }
    x_dict = {"company": data["company"].x}
    z_dict = model.encode(x_dict, full_edges_dict)
    embeddings = z_dict["company"].numpy()
    print(f"Embedding matrix: {embeddings.shape}")

    # Map deal_ids to acquirer embeddings
    ticker_to_id = meta["ticker_to_id"]
    deal_to_acq_ticker = meta["deal_to_acq_ticker"]

    records = []
    deals_found = 0
    deals_missing = 0

    for deal_id_str, acq_ticker in deal_to_acq_ticker.items():
        deal_id = int(deal_id_str)
        node_id = ticker_to_id.get(acq_ticker)

        if node_id is not None:
            emb = embeddings[node_id]
            record = {"deal_id": deal_id}
            for i in range(EMBED_DIM):
                record[f"graph_emb_{i}"] = float(emb[i])
            records.append(record)
            deals_found += 1
        else:
            deals_missing += 1

    df = pd.DataFrame(records)
    df = df.sort_values("deal_id").reset_index(drop=True)

    print(f"\nDeals with embeddings: {deals_found:,}")
    print(f"Deals missing (no node): {deals_missing}")
    print(f"Embedding dimensions: {EMBED_DIM}")

    emb_cols = [c for c in df.columns if c.startswith("graph_emb_")]
    emb_vals = df[emb_cols].values
    print(f"\nEmbedding stats:")
    print(f"  Mean: {emb_vals.mean():.4f}")
    print(f"  Std:  {emb_vals.std():.4f}")
    print(f"  Min:  {emb_vals.min():.4f}")
    print(f"  Max:  {emb_vals.max():.4f}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
