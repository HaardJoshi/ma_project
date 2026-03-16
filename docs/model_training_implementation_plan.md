# Model Training, Evaluation & Testing Plan

## Lit Review Alignment Check

> [!IMPORTANT]
> Your methodology chapter (cn6000.pdf §2.1) specifies a **MLP fusion architecture** — not pure XGBoost. The paper describes: *"The three distinct embeddings are concatenated into Z = [h_struct ∥ h_text ∥ h_fin]. This vector is passed through a Multi-Layer Perceptron (MLP) with ReLU and Dropout layers."* We **must** include the MLP to match your written methodology, while also using XGBoost/Ridge as baselines for comparison.

> [!WARNING]
> The methodology also mentions **Huber loss** as a sensitivity alternative to MSE, and **L2 regularization** on parameters. Both are straightforward to implement but must be included.

### Discrepancies Found & Resolutions

| Lit Review Says | Current Pipeline | Resolution |
|---|---|---|
| MLP fusion with Dropout + ReLU | XGBoost planned as primary | **Use both**: XGBoost as baseline, MLP as the primary model per methodology |
| Heterogeneous GNN (edge types) | GraphSAGE (homogeneous) | Minor gap — our graph has edge_type labels (supplier/customer). Can address in write-up; the embeddings already encode this |
| MSE + L2 with Huber sensitivity | Not yet specified | Implement MSE+L2 as primary loss, Huber as ablation |
| Financial ratios via linear projection | Raw concatenation | Add a learned linear projection layer for financial features in the MLP |

---

## Data Subsetting Strategy

Using **2,112 deals** that have CAR + Financials + SPLC graph data. ~45% also have text embeddings.

| Model | Features | Dims |
|---|---|---|
| M1: Financial only (baseline) | 67 financial ratios | 67 |
| M2: Financial + Text | 67 financial + 128 text PCA | 195 |
| M3: Financial + Text + Graph (full) | 67 financial + 128 text + 64 graph | 259 |

Target: `car_m5_p5` (11-day CAR) — continuous regression.

---

## Data Preprocessing Pipeline

### 1. Outlier Treatment — Winsorization
- **What:** Cap extreme values at the 1st and 99th percentile (per feature)
- **Why:** CAR and financial ratios have heavy tails. Extreme M&A deals (e.g., mega-mergers) can dominate training. Winsorization is standard in financial econometrics — preserves all observations while limiting influence of outliers
- **Applied to:** All financial features + CAR target
- **NOT applied to:** Embedding features (already bounded by FinBERT/GraphSAGE)

### 2. Feature Scaling — StandardScaler
- **What:** Zero-mean, unit-variance normalization: `x' = (x - μ) / σ`
- **Why:** Financial features span wildly different scales (Market Cap in billions vs ratios near 0-1). Without scaling, large-magnitude features dominate gradient updates in the MLP and distance calculations in Ridge
- **Applied to:** All features (financial + embeddings)
- **Fit on:** Training fold only (prevents data leakage)

### 3. Missing Value Handling
- **Financial features:** Median imputation (robust to outliers). Fit imputer on training fold only
- **Text embeddings:** Zero-fill for deals without EDGAR filings (45% of the 2,112 deals). Add binary indicator `has_text_features`
- **Graph embeddings:** All 2,112 deals have SPLC data — no missing values

### 4. Target Variable Treatment
- **Winsorize** CAR at 1st/99th percentile to limit extreme returns
- **No log transform** — CAR is already a percentage return centered near zero

---

## Model Architecture

### Tier 1: Linear Baselines

**Ridge Regression**
- Purpose: Establish linear baseline — if Ridge matches XGBoost, the relationship is linear
- L2 regularization naturally handles the 259:2,112 feature:sample ratio
- Hyperparameter: α (regularization strength) via cross-validation

**ElasticNet**
- Purpose: Feature selection baseline — L1 component drives irrelevant features to zero
- Hyperparameters: α (strength) + l1_ratio (L1/L2 mix)

### Tier 2: Tree-Based

**XGBoost**
- Purpose: Strong non-linear baseline — commonly used in financial ML literature
- Handles mixed feature types without explicit scaling
- Key hyperparameters (tuned via CV):
  - `max_depth=4-6` (prevent overfitting on 2,112 samples)
  - `min_child_weight=10-20` (require sufficient samples per leaf)
  - `learning_rate=0.01-0.1`
  - `n_estimators=200-500`
  - `reg_alpha` (L1) + `reg_lambda` (L2) regularization
  - `subsample=0.7-0.9` (row subsampling — similar purpose to dropout)
  - `colsample_bytree=0.7-0.9` (column subsampling per tree)

### Tier 3: MLP Fusion (Primary — matches lit review methodology)

**Architecture** (per cn6000.pdf §2.1.2):
```
Financial (67)  → Linear(67 → 32) → ReLU  → h_fin    (32)
Text (128)      → identity                 → h_text   (128)
Graph (64)      → identity                 → h_struct (64)
                                              ↓
                              Concat [h_fin, h_text, h_struct] = 224
                                              ↓
                              Linear(224 → 128) → ReLU → Dropout(0.3)
                              Linear(128 → 64)  → ReLU → Dropout(0.3)
                              Linear(64 → 1)    → CAR prediction
```

---

## Regularization & Anti-Overfitting Techniques

### Dropout (the technique you mentioned)
- **What:** During each training step, randomly zero out 30% of neurons in the MLP hidden layers
- **Why:** Forces the network to learn redundant representations. Prevents any single neuron from becoming a "result-dominating feature" — the network can't rely on any one pathway
- **Where:** After each hidden layer in the MLP (standard practice)
- **At inference:** Dropout is turned off; all neurons are active

### L2 Regularization (Weight Decay)
- **What:** Add λ·‖θ‖² penalty to the loss function
- **Why:** Penalizes large weights, preventing the model from fitting noise
- **Where:** All MLP parameters, Ridge regression, XGBoost `reg_lambda`
- **Lit review:** Explicitly specified in §2.1.1: ℒ(θ) = MSE + λ‖θ‖²

### Early Stopping
- **What:** Monitor validation loss during training; stop when it stops improving for N epochs
- **Why:** Prevents the MLP from memorizing training data
- **Patience:** 15-20 epochs (look-back window)

### Batch Normalization (optional)
- **What:** Normalize activations within each mini-batch
- **Why:** Stabilizes training, acts as mild regularizer
- **Where:** Between Linear and ReLU layers in MLP (if needed)

### Feature Subsampling (XGBoost)
- `subsample=0.8` — train each tree on 80% of rows (similar to dropout for trees)
- `colsample_bytree=0.8` — each tree sees 80% of features

---

## Evaluation Framework

### Cross-Validation
- **5-fold cross-validation** (stratified by CAR quantile to ensure each fold has balanced target distribution)
- Each fold: 1,690 train / 422 test
- **All preprocessing** (winsorization, scaling, imputation) fit on training fold only

### Metrics (per fold)
| Metric | Purpose |
|---|---|
| **MSE** | Primary loss (matches §2.1.1) |
| **RMSE** | Interpretable scale (% return) |
| **MAE** | Robust to outliers |
| **R²** | Coefficient of determination — H1 tests for R² improvement |
| **Huber loss** | Sensitivity analysis (§2.1.1 mentions this) |

### Statistical Significance Testing
- **Diebold-Mariano test** or **paired t-test** across CV folds to determine if Model M3 (full) significantly outperforms M1 (baseline) at p < 0.05 — required for H1
- **Feature importance:** SHAP values (XGBoost) and gradient-based attribution (MLP) to identify which modality contributes most

### Hypothesis-Specific Tests
- **H1 (Topological Alpha):** Compare R² of M1 vs M3, segmented by sector (supply-chain-dependent vs asset-light)
- **H2 (Semantic Divergence):** Compute cosine similarities between acquirer-target MD&A and Risk Factor embeddings, regress against CAR with separate coefficients
- **H3 (Topological Arbitrage):** Compute betweenness centrality and clustering coefficients from the supply chain graph, correlate with CAR variance

---

## Training Pipeline Steps

### Phase 1: Data Preparation
- [ ] Filter to 2,112 deals (CAR + Financials + SPLC)
- [ ] Winsorize financial features and CAR at 1st/99th percentile
- [ ] Create 5-fold CV splits (stratified by CAR quantile)
- [ ] Build preprocessing pipeline (imputer + scaler, fit on train only)

### Phase 2: Baseline Models
- [ ] Train Ridge regression (3 feature configs: M1, M2, M3)
- [ ] Train ElasticNet (3 configs)
- [ ] Train XGBoost with hyperparameter tuning (3 configs)
- [ ] Record all metrics per fold

### Phase 3: MLP Fusion Model
- [ ] Implement MLP fusion architecture (per §2.1.2)
- [ ] Train with MSE + L2 loss, Adam optimizer, early stopping
- [ ] Dropout = 0.3, lr = 1e-3, batch_size = 64
- [ ] Huber loss sensitivity run
- [ ] Record all metrics per fold

### Phase 4: Evaluation & Hypothesis Testing
- [ ] Compile cross-fold results table
- [ ] Run statistical significance tests (M1 vs M2 vs M3)
- [ ] H1: Sector-segmented R² comparison
- [ ] H2: Cosine similarity regression analysis
- [ ] H3: Centrality vs CAR variance analysis
- [ ] Generate SHAP/feature importance plots

### Phase 5: Documentation
- [ ] Results tables and comparison plots
- [ ] Update engineering log
- [ ] Commit and push all training code
