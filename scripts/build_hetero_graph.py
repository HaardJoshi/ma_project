"""
build_hetero_graph.py  --  Build PyG HeteroData supply chain graph
================================================================================
Constructs a heterogeneous supply chain graph with:
  - Node type: "company" (all firms)
  - Edge types: ("company", "supplies", "company") and ("company", "buys_from", "company")
  - Node features: 10 financial metrics (acquirers) or zeros (others)
  - Edge weights: revenue_pct / 100

This is the heterogeneous version matching the methodology (§2.1.2) —
separate message-passing per edge type.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
import json

# ── CONFIG ──────────────────────────────────────────────────────────────────
SPLC_FILE   = "data/interim/splc_full_data.csv"
DEALS_FILE  = "data/processed/final_car_dataset.csv"
MASTER_FILE = "data/interim/deals_master.csv"
OUTPUT_GRAPH = "data/interim/hetero_supply_chain_graph.pt"
OUTPUT_META  = "data/interim/hetero_graph_metadata.json"

# Financial features for node initialization
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

EMBEDDING_DIM = len(NODE_FEATURE_COLS)


def main():
    print("=" * 60)
    print("  BUILDING HETEROGENEOUS SUPPLY CHAIN GRAPH")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    splc = pd.read_csv(SPLC_FILE)
    deals = pd.read_csv(DEALS_FILE)
    master = pd.read_csv(MASTER_FILE)

    print(f"SPLC records:  {len(splc):,}")
    print(f"Deals:         {len(deals):,}")

    # ── Build ticker → node_id mapping ───────────────────────────
    acquirer_tickers = set()
    for _, row in master.iterrows():
        ticker = str(row["acq_ticker_bbg"]).strip()
        if ticker and ticker != "nan":
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
    acq_features = {}
    for idx, row in deals.iterrows():
        did = idx
        ticker = deal_to_acq_ticker.get(did)
        if ticker and ticker in ticker_to_id:
            feats = []
            for col in NODE_FEATURE_COLS:
                val = row.get(col, np.nan)
                feats.append(float(val) if pd.notna(val) else np.nan)
            if ticker not in acq_features or np.isnan(acq_features[ticker]).sum() > np.isnan(feats).sum():
                acq_features[ticker] = feats

    raw_features = np.zeros((num_nodes, EMBEDDING_DIM), dtype=np.float32)
    has_features = np.zeros(num_nodes, dtype=bool)

    for ticker, feats in acq_features.items():
        nid = ticker_to_id[ticker]
        raw_features[nid] = feats
        has_features[nid] = True

    print(f"\nNodes with financial features: {has_features.sum():,}")

    # Impute and scale
    col_means = np.nanmean(raw_features[has_features], axis=0)
    for j in range(EMBEDDING_DIM):
        if np.isnan(col_means[j]):
            col_means[j] = 0.0

    for i in range(num_nodes):
        for j in range(EMBEDDING_DIM):
            if np.isnan(raw_features[i, j]):
                raw_features[i, j] = col_means[j] if has_features[i] else 0.0

    scaler = StandardScaler()
    scaler.fit(raw_features[has_features])
    scaled_features = scaler.transform(raw_features)
    scaled_features[~has_features] = 0.0

    x = torch.tensor(scaled_features, dtype=torch.float32)

    # ── Build HETEROGENEOUS edges ────────────────────────────────
    # Two separate edge types:
    #   ("company", "supplies", "company")  — supplier → acquirer
    #   ("company", "buys_from", "company") — acquirer → customer
    supply_src, supply_dst, supply_weights = [], [], []
    buys_src, buys_dst, buys_weights = [], [], []
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
            supply_src.append(ent_id)
            supply_dst.append(acq_id)
            supply_weights.append(weight)
        else:
            buys_src.append(acq_id)
            buys_dst.append(ent_id)
            buys_weights.append(weight)

    print(f"\nEdges before dedup:")
    print(f"  supplies:  {len(supply_src):,}")
    print(f"  buys_from: {len(buys_src):,}")
    print(f"  skipped:   {skipped}")

    # Deduplicate each edge type separately (keep highest weight)
    def dedup_edges(src, dst, weights):
        edge_dict = {}
        for i in range(len(src)):
            key = (src[i], dst[i])
            if key not in edge_dict or weights[i] > edge_dict[key]:
                edge_dict[key] = weights[i]
        s, d, w = [], [], []
        for (si, di), wi in edge_dict.items():
            s.append(si)
            d.append(di)
            w.append(wi)
        return s, d, w

    supply_src, supply_dst, supply_weights = dedup_edges(supply_src, supply_dst, supply_weights)
    buys_src, buys_dst, buys_weights = dedup_edges(buys_src, buys_dst, buys_weights)

    print(f"\nAfter dedup:")
    print(f"  supplies:  {len(supply_src):,}")
    print(f"  buys_from: {len(buys_src):,}")
    print(f"  total:     {len(supply_src) + len(buys_src):,}")

    # ── Create PyG HeteroData object ─────────────────────────────
    data = HeteroData()

    # Single node type with features
    data["company"].x = x
    data["company"].num_nodes = num_nodes

    # Edge type 1: supplier → acquirer
    data["company", "supplies", "company"].edge_index = torch.tensor(
        [supply_src, supply_dst], dtype=torch.long
    )
    data["company", "supplies", "company"].edge_attr = torch.tensor(
        supply_weights, dtype=torch.float32
    ).unsqueeze(1)

    # Edge type 2: acquirer → customer
    data["company", "buys_from", "company"].edge_index = torch.tensor(
        [buys_src, buys_dst], dtype=torch.long
    )
    data["company", "buys_from", "company"].edge_attr = torch.tensor(
        buys_weights, dtype=torch.float32
    ).unsqueeze(1)

    print(f"\n{'='*60}")
    print(f"HETEROGENEOUS GRAPH SUMMARY:")
    print(f"  Node type:    'company' ({data['company'].num_nodes:,} nodes)")
    print(f"  Edge type 1:  'supplies' ({data['company', 'supplies', 'company'].edge_index.size(1):,} edges)")
    print(f"  Edge type 2:  'buys_from' ({data['company', 'buys_from', 'company'].edge_index.size(1):,} edges)")
    print(f"  Node features: {data['company'].x.shape[1]}")

    # ── Save ─────────────────────────────────────────────────────
    torch.save(data, OUTPUT_GRAPH)
    print(f"\nSaved graph to {OUTPUT_GRAPH}")

    metadata = {
        "ticker_to_id": ticker_to_id,
        "deal_to_acq_ticker": {str(k): v for k, v in deal_to_acq_ticker.items()},
        "node_feature_cols": NODE_FEATURE_COLS,
        "num_nodes": num_nodes,
        "num_supply_edges": len(supply_src),
        "num_buys_edges": len(buys_src),
        "edge_types": ["supplies", "buys_from"],
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {OUTPUT_META}")
    print("=" * 60)


if __name__ == "__main__":
    main()
