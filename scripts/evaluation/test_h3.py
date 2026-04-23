"""
test_h3.py  --  H3: Topological Arbitrage — Centrality vs CAR Variance
================================================================================
Tests if acquirers with high betweenness centrality (bridge nodes) exhibit
higher CAR variance, while those with high clustering coefficients
(embedded in dense clusters) show lower variance.

Loads the heterogeneous supply chain graph, computes centrality metrics
via NetworkX, and correlates with CAR outcomes.

Usage:
    python scripts/test_h3.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import networkx as nx
from scipy import stats

from training_utils import load_and_prepare_data, SEED, RESULTS_DIR

np.random.seed(SEED)


def main():
    print("=" * 70)
    print("  H3: TOPOLOGICAL ARBITRAGE — CENTRALITY vs CAR VARIANCE")
    print("=" * 70)

    subset, y_cont = load_and_prepare_data()

    # Load graph
    graph_file = "data/interim/hetero_supply_chain_graph.pt"
    meta_file = "data/interim/hetero_graph_metadata.json"

    if not os.path.exists(graph_file) or not os.path.exists(meta_file):
        print("  ❌ Graph files not found")
        return

    data = torch.load(graph_file, weights_only=False)
    with open(meta_file, "r") as f:
        meta = json.load(f)

    ticker_to_id = meta["ticker_to_id"]
    deal_to_acq_ticker = meta["deal_to_acq_ticker"]

    # Build NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(range(data["company"].num_nodes))

    for edge_type in [("company", "supplies", "company"), ("company", "buys_from", "company")]:
        ei = data[edge_type].edge_index
        for i in range(ei.size(1)):
            G.add_edge(ei[0, i].item(), ei[1, i].item())

    print(f"  NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Compute centrality
    print("  Computing betweenness centrality...")
    betweenness = nx.betweenness_centrality(G)
    print("  Computing clustering coefficients...")
    clustering = nx.clustering(G.to_undirected())
    print("  Computing degree centrality...")
    degree = nx.degree_centrality(G)

    # Map to deals
    records = []
    for deal_id_str, acq_ticker in deal_to_acq_ticker.items():
        deal_id = int(deal_id_str)
        node_id = ticker_to_id.get(acq_ticker)
        if node_id is not None:
            records.append({
                "deal_id": deal_id,
                "betweenness": betweenness.get(node_id, 0),
                "clustering": clustering.get(node_id, 0),
                "degree": degree.get(node_id, 0),
            })

    centrality_df = pd.DataFrame(records)
    print(f"  Deals with centrality metrics: {len(centrality_df)}")

    # Merge with CAR
    subset_copy = subset.copy()
    subset_copy["deal_id"] = subset_copy.index
    subset_copy["car"] = y_cont
    merged = subset_copy.merge(centrality_df, on="deal_id", how="inner")
    merged["abs_car"] = merged["car"].abs()

    print(f"  Merged deals: {len(merged)}")

    # ── Descriptive stats ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  CENTRALITY DESCRIPTIVE STATISTICS")
    print(f"{'─'*70}")

    for metric in ["betweenness", "clustering", "degree"]:
        vals = merged[metric]
        nonzero = (vals > 0).sum()
        print(f"  {metric:14s}: mean={vals.mean():.6f}, std={vals.std():.6f}, "
              f"nonzero={nonzero} ({100*nonzero/len(vals):.1f}%)")

    # ── Quartile analysis ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  QUARTILE ANALYSIS: CENTRALITY vs CAR")
    print(f"{'─'*70}")

    for metric in ["betweenness", "clustering", "degree"]:
        # Use rank-based quartiles to handle ties at zero
        ranks = merged[metric].rank(method="first")
        q_labels = pd.qcut(ranks, q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])

        print(f"\n  {metric.upper()}:")
        q_stats = []
        for q in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
            mask = q_labels == q
            car = merged.loc[mask, "car"]
            abs_car = merged.loc[mask, "abs_car"]
            q_stats.append({
                "q": q, "n": mask.sum(),
                "car_mean": car.mean(), "car_std": car.std(),
                "car_var": car.var(), "abs_car_mean": abs_car.mean()
            })
            print(f"    {q:10s}: n={mask.sum():4d} | "
                  f"CAR mean={car.mean():+.5f} | "
                  f"std={car.std():.5f} | "
                  f"|CAR| mean={abs_car.mean():.5f}")

        # Q4 vs Q1 variance comparison (Levene test)
        q1_mask = q_labels == "Q1 (low)"
        q4_mask = q_labels == "Q4 (high)"
        lev_stat, lev_p = stats.levene(
            merged.loc[q1_mask, "car"], merged.loc[q4_mask, "car"]
        )
        print(f"    Levene test (Q4 vs Q1 variance): F={lev_stat:.3f}, p={lev_p:.4f} "
              f"{'✅ sig' if lev_p<0.05 else '❌ n.s.'}")

    # ── Correlations ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  CORRELATIONS")
    print(f"{'─'*70}")

    for metric in ["betweenness", "clustering", "degree"]:
        r_car, p_car = stats.pearsonr(merged[metric], merged["car"])
        r_abs, p_abs = stats.pearsonr(merged[metric], merged["abs_car"])
        r_var, p_var = stats.spearmanr(merged[metric], merged["abs_car"])

        print(f"  {metric:14s} vs CAR:     r={r_car:+.4f}, p={p_car:.4f} "
              f"{'✅' if p_car<0.05 else '❌'}")
        print(f"  {metric:14s} vs |CAR|:   r={r_abs:+.4f}, p={p_abs:.4f} "
              f"{'✅' if p_abs<0.05 else '❌'} (Pearson)")
        print(f"  {metric:14s} vs |CAR|:   ρ={r_var:+.4f}, p={p_var:.4f} "
              f"{'✅' if p_var<0.05 else '❌'} (Spearman)")
        print()

    # ── H3 conclusion ──────────────────────────────────────────
    r_bet, p_bet = stats.pearsonr(merged["betweenness"], merged["abs_car"])
    r_clu, p_clu = stats.pearsonr(merged["clustering"], merged["abs_car"])

    print(f"{'='*70}")
    print("  H3 CONCLUSION")
    print(f"{'='*70}")
    print(f"  Betweenness → |CAR|: r={r_bet:+.4f} (expected: positive)")
    print(f"  Clustering → |CAR|:  r={r_clu:+.4f} (expected: negative)")

    h3_bet = r_bet > 0
    h3_clu = r_clu < 0
    if h3_bet and h3_clu:
        print(f"\n  ✅ H3 SUPPORTED: Bridge nodes → higher variance, clustered → lower")
    elif h3_bet or h3_clu:
        print(f"\n  ⚠️ H3 PARTIALLY SUPPORTED:")
        print(f"    Betweenness direction: {'✅ correct' if h3_bet else '❌ opposite'}")
        print(f"    Clustering direction:  {'✅ correct' if h3_clu else '❌ opposite'}")
    else:
        print(f"\n  ❌ H3 NOT SUPPORTED")

    print("=" * 70)


if __name__ == "__main__":
    main()
