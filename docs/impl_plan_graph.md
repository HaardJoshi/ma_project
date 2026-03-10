# Graph Embedding Extraction Pipeline

Extract 64-dimensional supply chain graph embeddings for each M&A deal using a self-supervised GraphSAGE model trained on link prediction.

## User Review Required

> [!IMPORTANT]
> **Dependencies:** PyTorch, PyG (torch-geometric), NetworkX, and Scikit-learn are **not installed**. We will install them first. PyTorch will be CPU-only (no CUDA on this machine).

> [!WARNING]
> **Training time:** Self-supervised link prediction on ~6K nodes and ~18K edges is small by GNN standards. Training should take < 5 minutes on CPU.

## Proposed Changes

### Step 1: Install Dependencies

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric networkx scikit-learn
```

---

### Step 2: Build the Graph

#### [NEW] [build_graph.py](file:///c:/Users/u2512658/hardjoshi-ma/scripts/build_graph.py)

- Load `splc_full_data.csv` and [final_car_dataset.csv](file:///c:/Users/u2512658/hardjoshi-ma/data/processed/final_car_dataset.csv)
- Create a **global heterogeneous graph** with all unique companies as nodes
- Map each unique Bloomberg ticker → integer node ID
- Create directed edges: `supplier → acquirer` and `acquirer → customer`
- Edge weights = `revenue_pct / 100` (normalized to [0,1])
- Node features for acquirers: 10-15 key financial metrics (scaled with StandardScaler)
- Nodes without financial data (suppliers/customers that aren't acquirers): get zero-vectors or mean-imputed values
- Save the PyG `Data` object to `data/interim/supply_chain_graph.pt`

**Key financial features for node initialization:**
| Feature | Column |
|---|---|
| Market Cap | `Acquirer Current Market Cap` |
| Total Assets | `Acquirer Total Assets` |
| Revenue | `Acquirer Sales/Revenue/Turnover` |
| EBITDA | `Acquirer EBITDA(...)` |
| Operating Margin | `Acquirer Operating Margin` |
| P/E Ratio | `Acquirer Price Earnings Ratio (P/E)` |
| Debt/Assets | `Acquirer Total Debt to Total Assets` |
| Current Ratio | `Acquirer Current Ratio` |
| ROE | `Acquirer Return on Common Equity` |
| Revenue Growth | `Acquirer Net Revenue Growth` |

---

### Step 3: Train GNN & Extract Embeddings

#### [NEW] [train_graph_embeddings.py](file:///c:/Users/u2512658/hardjoshi-ma/scripts/train_graph_embeddings.py)

**Model Architecture:**
```
GraphSAGE(
  conv1: SAGEConv(in_features → 128, aggr='mean')
  conv2: SAGEConv(128 → 64, aggr='mean')
)
```

**Self-supervised training (Link Prediction):**
1. Randomly mask 15% of edges as positive test samples
2. Generate equal number of negative (non-existent) edges
3. Train the GNN to predict which edges exist vs don't
4. Loss: Binary Cross-Entropy on dot-product similarity of node embeddings
5. Train for 200 epochs, Adam optimizer, lr=0.01

**After training:**
1. Run forward pass on full graph → get 64-dim embedding for every node
2. For each deal, look up the acquirer's node → extract its 64-dim embedding
3. Save to `data/interim/graph_embeddings.csv` with columns: `deal_id, graph_emb_0, ..., graph_emb_63`

---

### Step 4: Merge into Final Dataset

#### [NEW] [merge_graph_embeddings.py](file:///c:/Users/u2512658/hardjoshi-ma/scripts/merge_graph_embeddings.py)

- Join `graph_embeddings.csv` with [final_car_dataset.csv](file:///c:/Users/u2512658/hardjoshi-ma/data/processed/final_car_dataset.csv) on `deal_id`
- Output: `data/processed/final_multimodal_dataset.csv`
  - Block A: 60+ financial features
  - Block B: 128 text PCA features (64 MD&A + 64 RF)
  - Block C: 64 graph embedding features ← **NEW**
  - Target: `car_m5_p5`

## Verification Plan

### Automated Tests
- Verify graph has expected node/edge counts
- Verify link prediction AUC > 0.7 (proves the GNN learned meaningful structure)
- Verify embedding dimensions = 64 per deal
- Verify final merged dataset row count = 4,999

### Manual Verification
- Spot-check that similar companies (e.g., two telecom acquirers) have similar embeddings via cosine similarity
