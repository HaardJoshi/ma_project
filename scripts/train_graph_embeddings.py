"""
train_graph_embeddings.py  --  Train GraphSAGE & extract node embeddings
================================================================================
Trains a 2-layer GraphSAGE model using self-supervised link prediction on the
supply chain graph. After training, extracts 64-dim embeddings for each acquirer
node and maps them back to deal_ids.

Output: data/interim/graph_embeddings.csv
"""

import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

# ── CONFIG ──────────────────────────────────────────────────────────────────
GRAPH_FILE  = "data/interim/supply_chain_graph.pt"
META_FILE   = "data/interim/graph_metadata.json"
OUTPUT_CSV  = "data/interim/graph_embeddings.csv"

HIDDEN_DIM  = 128
EMBED_DIM   = 64
EPOCHS      = 200
LR          = 0.01
TEST_RATIO  = 0.15  # fraction of edges held out for validation
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── MODEL ────────────────────────────────────────────────────────────────────
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def encode(self, x, edge_index):
        """Get node embeddings (no dropout)."""
        self.eval()
        with torch.no_grad():
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
        return x


def decode(z, edge_index):
    """Predict edge likelihood via dot product of node embeddings."""
    src, dst = edge_index
    return (z[src] * z[dst]).sum(dim=1)


def train_epoch(model, optimizer, data, train_edge_index):
    model.train()
    optimizer.zero_grad()

    # Forward pass with training edges only
    z = model(data.x, train_edge_index)

    # Positive edges (existing)
    pos_pred = decode(z, train_edge_index)

    # Negative edges (non-existing)
    neg_edge_index = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=train_edge_index.size(1),
    )
    neg_pred = decode(z, neg_edge_index)

    # Binary cross-entropy loss
    pos_labels = torch.ones(pos_pred.size(0))
    neg_labels = torch.zeros(neg_pred.size(0))

    preds = torch.cat([pos_pred, neg_pred])
    labels = torch.cat([pos_labels, neg_labels])

    loss = F.binary_cross_entropy_with_logits(preds, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, train_edge_index, test_edge_index):
    model.eval()
    z = model(data.x, train_edge_index)

    # Positive test edges
    pos_pred = decode(z, test_edge_index).sigmoid().cpu().numpy()

    # Negative test edges
    neg_edge_index = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=test_edge_index.size(1),
    )
    neg_pred = decode(z, neg_edge_index).sigmoid().cpu().numpy()

    labels = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
    preds = np.concatenate([pos_pred, neg_pred])

    auc = roc_auc_score(labels, preds)
    return auc


def main():
    print("=" * 60)
    print("  TRAINING GRAPHSAGE FOR LINK PREDICTION")
    print("=" * 60)

    # Load graph and metadata
    data = torch.load(GRAPH_FILE, weights_only=False)
    with open(META_FILE, "r") as f:
        meta = json.load(f)

    print(f"Graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    print(f"Node features: {data.x.shape[1]}")

    # ── Train/test edge split ────────────────────────────────────
    num_edges = data.edge_index.size(1)
    perm = torch.randperm(num_edges)
    n_test = int(num_edges * TEST_RATIO)

    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    train_edge_index = data.edge_index[:, train_idx]
    test_edge_index = data.edge_index[:, test_idx]

    print(f"Train edges: {train_edge_index.size(1):,}")
    print(f"Test edges:  {test_edge_index.size(1):,}")

    # ── Initialize model ─────────────────────────────────────────
    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_DIM,
        out_channels=EMBED_DIM,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nModel: GraphSAGE({data.x.shape[1]} -> {HIDDEN_DIM} -> {EMBED_DIM})")
    print(f"Training for {EPOCHS} epochs...")

    # ── Training loop ────────────────────────────────────────────
    best_auc = 0
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, optimizer, data, train_edge_index)

        if epoch % 20 == 0 or epoch == 1:
            auc = evaluate(model, data, train_edge_index, test_edge_index)
            if auc > best_auc:
                best_auc = auc
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Test AUC: {auc:.4f}")

    print(f"\nBest test AUC: {best_auc:.4f}")

    # ── Extract embeddings ───────────────────────────────────────
    print("\nExtracting node embeddings...")
    embeddings = model.encode(data.x, data.edge_index)
    embeddings = embeddings.numpy()
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

    # Quick sanity check: embedding variance
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
