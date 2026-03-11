# Phase 3: Hypothesis Testing — Checklist

## H1: Sector-Segmented Topological Alpha (~30 min)
- [ ] Define sector groups from SIC codes:
  - Supply-chain-dependent: Manufacturing (20-39), Transport (40-49)
  - Asset-light: Finance (60-67), Tech Services (73), Business Services (70-79)
- [ ] Run XGBoost classifier M1 vs M3 within each sector group
- [ ] Compare AUC improvement Δ(M3-M1) across groups
- [ ] Test: does graph data help MORE in supply-chain sectors?
- [ ] Document findings

## H2: Semantic Divergence (~1 hour)
- [ ] Compute cosine similarity between acquirer and target MD&A PCA embeddings
- [ ] Compute cosine similarity between acquirer and target RF PCA embeddings
- [ ] Regress CAR against both similarities (separate coefficients)
- [ ] Test: MD&A similarity positive AND RF similarity negative?
- [ ] Visualise: scatter plot of similarity vs CAR per section
- [ ] Document findings

## H3: Topological Arbitrage (~30 min)
- [ ] Compute betweenness centrality for all nodes in supply chain graph
- [ ] Compute clustering coefficient for all nodes
- [ ] Map acquirer centrality metrics to deals
- [ ] Correlate with CAR mean and CAR variance
- [ ] Test: high betweenness → higher CAR variance?
- [ ] Document findings

## Wrap-Up
- [ ] Consolidated Phase 3 results document
- [ ] Update dissertation notes with H1/H2/H3 conclusions
- [ ] Commit and push
