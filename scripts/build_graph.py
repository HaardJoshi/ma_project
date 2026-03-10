"""
build_graph.py  --  Build PyG supply chain graph from SPLC data
================================================================================
Constructs a global supply chain graph where:
  - Nodes = all unique companies (acquirers + suppliers + customers)
  - Edges = directed supply chain relationships
  - Edge weights = revenue_pct (normalized)
  - Node features = financial metrics (for acquirers) or zeros (for others)

Saves the graph as a PyG Data object for GNN training.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import json

# ── CONFIG ──────────────────────────────────────────────────────────────────
SPLC_FILE   = "data/interim/splc_full_data.csv"
DEALS_FILE  = "data/processed/final_car_dataset.csv"
MASTER_FILE = "data/interim/deals_master.csv"
OUTPUT_GRAPH = "data/interim/supply_chain_graph.pt"
OUTPUT_META  = "data/interim/graph_metadata.json"

# Financial features to use as node features (acquirer-level)
NODE_FEATURE_COLS = [
    "Acquirer Current Market Cap",
    "Acquirer Total Assets",
    "Acquirer Sales/Revenue/Turnover",
    "Acquirer EBITDA(Earn Bef Int Dep & Amo)",
    "Acquirer Operating Margin",
    "Acquirer Price Earnings Ratio (P/E)",
    "Acquirer Total Debt to Total Assets",
    "Acquirer Current Ratio",
    "Acquirer Return on Common Equity",
    "Acquirer Net Revenue Growth",
]

EMBEDDING_DIM = len(NODE_FEATURE_COLS)  # 10 features


def main():
    print("=" * 60)
    print("  BUILDING SUPPLY CHAIN GRAPH")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    splc = pd.read_csv(SPLC_FILE)
    deals = pd.read_csv(DEALS_FILE)
    master = pd.read_csv(MASTER_FILE)

    print(f"SPLC records:  {len(splc):,}")
    print(f"Deals:         {len(deals):,}")

    # ── Build ticker → node_id mapping ───────────────────────────
    # Collect all unique tickers (acquirers + supply chain entities)
    acquirer_tickers = set()
    for _, row in master.iterrows():
        ticker = str(row["acq_ticker_bbg"]).strip()
        if ticker and ticker != "nan":
            # Normalize: add " Equity" if not present for consistency
            if not ticker.endswith("Equity"):
                ticker = ticker + " Equity"
            acquirer_tickers.add(ticker)

    entity_tickers = set(splc["entity_ticker"].dropna().unique())
    all_tickers = sorted(acquirer_tickers | entity_tickers)

    ticker_to_id = {t: i for i, t in enumerate(all_tickers)}
    num_nodes = len(all_tickers)

    print(f"\nUnique companies (nodes): {num_nodes:,}")
    print(f"  Acquirers:   {len(acquirer_tickers):,}")
    print(f"  SC entities: {len(entity_tickers):,}")
    print(f"  Overlap:     {len(acquirer_tickers & entity_tickers):,}")

    # ── Build deal_id → acquirer ticker mapping ──────────────────
    deal_to_acq_ticker = {}
    for _, row in master.iterrows():
        ticker = str(row["acq_ticker_bbg"]).strip()
        if ticker and ticker != "nan":
            if not ticker.endswith("Equity"):
                ticker = ticker + " Equity"
            deal_to_acq_ticker[row["deal_id"]] = ticker

    # ── Build node features ──────────────────────────────────────
    # Only acquirers have financial features; other nodes get zeros
    # First, build acquirer ticker → financial features mapping
    acq_features = {}
    for idx, row in deals.iterrows():
        did = idx  # deal_id is the row index
        ticker = deal_to_acq_ticker.get(did)
        if ticker and ticker in ticker_to_id:
            feats = []
            for col in NODE_FEATURE_COLS:
                val = row.get(col, np.nan)
                feats.append(float(val) if pd.notna(val) else np.nan)
            # If this ticker already has features, keep the one with fewer NaNs
            if ticker not in acq_features or np.isnan(acq_features[ticker]).sum() > np.isnan(feats).sum():
                acq_features[ticker] = feats

    # Build feature matrix
    raw_features = np.zeros((num_nodes, EMBEDDING_DIM), dtype=np.float32)
    has_features = np.zeros(num_nodes, dtype=bool)

    for ticker, feats in acq_features.items():
        nid = ticker_to_id[ticker]
        raw_features[nid] = feats
        has_features[nid] = True

    print(f"\nNodes with financial features: {has_features.sum():,}")

    # Impute NaN values with column mean, then scale
    col_means = np.nanmean(raw_features[has_features], axis=0)
    for j in range(EMBEDDING_DIM):
        if np.isnan(col_means[j]):
            col_means[j] = 0.0

    # Fill NaNs in featured nodes with column means
    for i in range(num_nodes):
        for j in range(EMBEDDING_DIM):
            if np.isnan(raw_features[i, j]):
                raw_features[i, j] = col_means[j] if has_features[i] else 0.0

    # StandardScale the features
    scaler = StandardScaler()
    # Fit on nodes that have real features
    scaler.fit(raw_features[has_features])
    # Transform all nodes
    scaled_features = scaler.transform(raw_features)
    # Zero out non-featured nodes (they should remain at zero, not at -mean/std)
    scaled_features[~has_features] = 0.0

    x = torch.tensor(scaled_features, dtype=torch.float32)

    # ── Build edges ──────────────────────────────────────────────
    src_list = []
    dst_list = []
    edge_weights = []
    edge_types = []  # 0 = supplier→acquirer, 1 = acquirer→customer
    skipped = 0

    for _, row in splc.iterrows():
        did = row["deal_id"]
        acq_ticker = deal_to_acq_ticker.get(did)
        entity_ticker = row["entity_ticker"]

        if acq_ticker is None or acq_ticker not in ticker_to_id:
            skipped += 1
            continue
        if entity_ticker not in ticker_to_id:
            skipped += 1
            continue

        acq_id = ticker_to_id[acq_ticker]
        ent_id = ticker_to_id[entity_ticker]

        rev_pct = row["revenue_pct"]
        weight = float(rev_pct) / 100.0 if pd.notna(rev_pct) and rev_pct > 0 else 0.01

        if row["role"] == "supplier":
            # Supplier → Acquirer
            src_list.append(ent_id)
            dst_list.append(acq_id)
            edge_types.append(0)
        else:
            # Acquirer → Customer
            src_list.append(acq_id)
            dst_list.append(ent_id)
            edge_types.append(1)

        edge_weights.append(weight)

    print(f"\nEdges created: {len(src_list):,}")
    print(f"Edges skipped (missing tickers): {skipped}")
    print(f"  Supplier->Acquirer: {sum(1 for t in edge_types if t == 0):,}")
    print(f"  Acquirer->Customer: {sum(1 for t in edge_types if t == 1):,}")

    # Remove duplicate edges (keep highest weight)
    edge_dict = {}
    for i in range(len(src_list)):
        key = (src_list[i], dst_list[i])
        if key not in edge_dict or edge_weights[i] > edge_dict[key][0]:
            edge_dict[key] = (edge_weights[i], edge_types[i])

    deduped_src = []
    deduped_dst = []
    deduped_weights = []
    deduped_types = []
    for (s, d), (w, t) in edge_dict.items():
        deduped_src.append(s)
        deduped_dst.append(d)
        deduped_weights.append(w)
        deduped_types.append(t)

    print(f"\nAfter dedup: {len(deduped_src):,} unique edges")

    edge_index = torch.tensor([deduped_src, deduped_dst], dtype=torch.long)
    edge_attr = torch.tensor(deduped_weights, dtype=torch.float32).unsqueeze(1)
    edge_type = torch.tensor(deduped_types, dtype=torch.long)

    # ── Create PyG Data object ───────────────────────────────────
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        num_nodes=num_nodes,
    )

    print(f"\n{'='*60}")
    print(f"GRAPH SUMMARY:")
    print(f"  Nodes:          {data.num_nodes:,}")
    print(f"  Edges:          {data.num_edges:,}")
    print(f"  Node features:  {data.x.shape[1]}")
    print(f"  Edge attr dim:  {data.edge_attr.shape[1]}")
    print(f"  Avg degree:     {data.num_edges / data.num_nodes:.2f}")

    # ── Save ─────────────────────────────────────────────────────
    torch.save(data, OUTPUT_GRAPH)
    print(f"\nSaved graph to {OUTPUT_GRAPH}")

    # Save metadata for later use
    metadata = {
        "ticker_to_id": ticker_to_id,
        "deal_to_acq_ticker": {str(k): v for k, v in deal_to_acq_ticker.items()},
        "node_feature_cols": NODE_FEATURE_COLS,
        "num_nodes": num_nodes,
        "num_edges": len(deduped_src),
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {OUTPUT_META}")
    print("=" * 60)


if __name__ == "__main__":
    main()
