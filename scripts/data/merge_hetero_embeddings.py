"""
merge_hetero_embeddings.py  --  Merge heterogeneous graph embeddings into final dataset
================================================================================
Replaces the old homogeneous graph embeddings in final_multimodal_dataset.csv
with the new heterogeneous GraphSAGE embeddings.

Output: data/processed/final_multimodal_dataset.csv (updated)
"""

import pandas as pd
import numpy as np

DEALS_FILE = "data/processed/final_car_dataset.csv"
GRAPH_FILE = "data/interim/hetero_graph_embeddings.csv"
OUTPUT_FILE = "data/processed/final_multimodal_dataset.csv"

print("=" * 60)
print("  MERGING HETEROGENEOUS GRAPH EMBEDDINGS")
print("=" * 60)

# Load datasets
deals = pd.read_csv(DEALS_FILE)
graph = pd.read_csv(GRAPH_FILE)

print(f"Deals:      {len(deals):,} rows, {deals.shape[1]} cols")
print(f"Graph embs: {len(graph):,} rows, {graph.shape[1]} cols")

# Merge on deal_id (row index)
# deal_id in graph_embeddings corresponds to DataFrame index in deals
graph = graph.drop_duplicates(subset=["deal_id"], keep="first")
deals_indexed = deals.copy()
deals_indexed["deal_id"] = deals_indexed.index

# Drop old graph embedding columns if they exist
old_graph_cols = [c for c in deals_indexed.columns if c.startswith("graph_emb_")]
if old_graph_cols:
    print(f"  Dropping {len(old_graph_cols)} old graph embedding columns")
    deals_indexed = deals_indexed.drop(columns=old_graph_cols)

combined = deals_indexed.merge(graph, on="deal_id", how="left")

# Fill missing graph embeddings with zeros
graph_cols = [c for c in combined.columns if c.startswith("graph_emb_")]
combined[graph_cols] = combined[graph_cols].fillna(0.0)

# Add has_graph indicator
combined["has_graph"] = combined["graph_emb_0"].ne(0).astype(int)

# Drop the temp deal_id col (it's just the index)
combined = combined.drop(columns=["deal_id"])

# Count coverage
has_graph = (combined[graph_cols].abs().sum(axis=1) > 0).sum()

print(f"\nCombined: {len(combined):,} rows, {combined.shape[1]} cols")
print(f"  With graph embeddings: {has_graph:,} ({100*has_graph/len(combined):.1f}%)")
print(f"  Without graph (zeros): {len(combined)-has_graph:,}")

# Column breakdown
fin_cols = [c for c in combined.columns if not c.startswith(("mda_pca_", "rf_pca_", "graph_emb_", "has_", "deal_key", "alpha_hat", "beta_hat", "car_", "n_est", "n_event"))]
text_cols = [c for c in combined.columns if c.startswith(("mda_pca_", "rf_pca_"))]
meta_cols = [c for c in combined.columns if c.startswith(("has_", "deal_key", "alpha_hat", "beta_hat", "car_", "n_est", "n_event"))]

print(f"\nColumn breakdown:")
print(f"  Financial features: {len(fin_cols)}")
print(f"  Text embeddings:    {len(text_cols)}")
print(f"  Graph embeddings:   {len(graph_cols)}")
print(f"  Metadata/target:    {len(meta_cols)}")

combined.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved to {OUTPUT_FILE}")
print("=" * 60)

if __name__ == "__main__":
    pass
