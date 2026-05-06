<div align="center">

# Breaking the Tabular Ceiling
## Heterogeneous Graph-Aware Multimodal Fusion for M&A Synergy Prediction

*BSc (Hons) Data Science & Artificial Intelligence — CN6000 Final Dissertation*
*University of East London · Supervisor: Arish Siddiqui · May 2026*

---

[![Python](https://img.shields.io/badge/Python-3.10+-4A90D9?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deal%20Intelligence%20Terminal-E8A838?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-HeteroGraphSAGE-2ECC71?style=flat-square)](https://pyg.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Fusion%20Classifier-E74C3C?style=flat-square)](https://xgboost.ai)
[![FinBERT](https://img.shields.io/badge/FinBERT-Section--Aware%20NLP-9B59B6?style=flat-square)](https://huggingface.co/ProsusAI/finbert)
[![License: MIT](https://img.shields.io/badge/License-MIT-white?style=flat-square)](LICENSE)

</div>

---

> **70–90% of M&A deals destroy shareholder value.** This dissertation argues the root cause is architectural, not computational: every prior quantitative model treats each firm as an isolated data point — blind to the supply-chain topology, competitive network structure, and section-level semantic signals that collectively determine whether a combination will create or destroy value.

---

## The Core Argument

Every generation of M&A prediction model — from 1980s logistic regression through XGBoost and transformer NLP — inherits the same fatal assumption: **independence**. A model that processes each deal as a row in a spreadsheet cannot know that the acquirer's largest customer is near bankruptcy, that acquirer and target share a fragile single-source supplier, or that the combined entity will create a critical bottleneck in an industrial network. None of those facts appear in a financial ratio vector.

This study breaks that constraint by fusing three information modalities that, together, cover what each paradigm individually missed:

```
Block A  ──  56 financial ratios          (the baseline every prior model used)
Block B  ──  Section-aware FinBERT text   (MD&A and Risk Factors processed separately)
Block C  ──  HeteroGraphSAGE topology     (supply-chain ecosystem embeddings)
             ↓  late concatenation
             z_i ∈ ℝ²⁴⁹   →   XGBoost Classifier   →   P(CAR > 0)
```

The three blocks are fused via late concatenation and evaluated under a **dual-pipeline framework** — a classifier predicting binary CAR direction, and a regressor attempting CAR magnitude. As Chapter 4 demonstrates, these resolve differently, and *that difference is itself a substantive finding*.

---

## Empirical Results at a Glance

All results derive from **5-fold stratified cross-validation** with strict temporal splits (2000–2016 train / 2017–2019 val / 2020–2023 test) and an **11-day event-window embargo** to prevent forward-looking leakage.

### Classification Ablation Ladder

| Model | Description | Features | AUC-ROC | Accuracy | F1 |
|-------|-------------|----------|---------|----------|-----|
| **M1** | Financial only (XGBoost baseline) | 56 | 0.5408 | 52.8% | 0.473 |
| **M2** | Financial + Pooled FinBERT text | 184 | 0.5289 ❌ | 52.9% | 0.476 |
| **M3** | Full Fusion (F + T + G) | 248 | **0.5655** ✅ | 54.8% | 0.490 |
| **M3e** | M3 + Auxiliary scalar features | 261 | 0.5585 | 55.1% | 0.492 |

> **The M2 Reversal** — the finding that naive NLP *degrades* prediction by −0.0119 AUC — is one of the study's most important results. Undifferentiated FinBERT embeddings introduce semantic noise. Section-aware processing (MD&A vs. Risk Factors separately) is what recovers and then *exceeds* the baseline.

### Regression Pipeline (Boundary Test)

| Model | R² | Interpretation |
|-------|-----|----------------|
| M1 Financial only | −0.008 | No explanatory power above sample mean |
| M2 Financial + Text | −0.155 | Naive text aggregation adds noise |
| M3 Full Fusion | −0.164 | Multimodal fusion helps direction, not magnitude |

> **Negative R² is not a failure — it is a finding.** It precisely locates the boundary between tractable (sign discrimination) and structurally intractable (point-magnitude prediction) M&A sub-problems.

---

## Three Formal Hypotheses (All Supported)

### H1 — Topological Alpha ✅
> *The inclusion of heterogeneous supply-chain graph topology yields a statistically significant improvement in directional deal discrimination over a financial-only baseline.*

**Result:** M3 achieves AUC = 0.5655 vs. M1 baseline 0.5408. Paired t-test confirms the +0.0247 gain is statistically significant. Sector-stratified analysis shows the gain is concentrated in supply-chain intensive industries (Energy, Industrials, Materials), consistent with the theoretical prediction.

### H2 — Semantic Divergence ✅
> *MD&A similarity (strategic fit) correlates positively with CAR; Risk Factor similarity (shared liability) correlates negatively. These opposing effects are mathematically suppressed when sections are pooled.*

**Result:** OLS on a semantic-divergence subsample of 1,140 deals recovers:
- β_MDA = **+0.0044** (95% CI: [+0.0023, +0.0065]) — strategic fit is positive
- β_RF  = **−0.0080** (95% CI: [−0.0101, −0.0059]) — shared liability is negative

Neither confidence interval crosses zero. The M2 reversal provides the direct experimental proof: pooling the two sections cancels the opposing signals and produces a net-negative contribution.

### H3 — Topological Arbitrage ✅
> *Acquirers with high betweenness centrality exhibit statistically compressed variance in announcement-return outcomes relative to peripheral acquirers.*

**Result:** Levene's test across betweenness-centrality quantile groups: **F = 7.07, p = 0.0079**. Structurally central acquirers are more predictable — their supply-chain visibility functions as an information transparency mechanism that dampens idiosyncratic return shocks.

---

## Architecture Deep Dive

### Feature Blocks

| Block | Source | Dimension | Encoding | Notes |
|-------|--------|-----------|----------|-------|
| **A — Financial** | Yahoo Finance / Bloomberg | 56 | StandardScaler → XGBoost | Leverage, liquidity, profitability, deal structure |
| **B — Textual** | SEC EDGAR 10-K | 128 (64+64) | Frozen FinBERT → PCA | MD&A and Risk Factors encoded *separately* |
| **C — Graph** | Bloomberg SPLC | 65 (64+1) | 2-hop HeteroGraphSAGE | Supply-chain + competitor edges; `has_graph` indicator |
| **Fused** | — | 249 | Late concatenation | z_i = [h_F ‖ h_T ‖ h_G] |

### HeteroGraphSAGE (Block C)

The supply-chain graph is heterogeneous: nodes are firms, edge types are `SUPPLIES_TO`, `COMPETES_WITH`, and `CUSTOMERS_OF`. The two-hop neighbourhood aggregation allows each firm's embedding to encode not just its direct relationships but the *relationships of its relationships* — the second-order industrial ecosystem that conventional models are architecturally blind to.

```python
# Edge type schema
edge_types = [
    ("firm", "SUPPLIES_TO",   "firm"),   # upstream dependency
    ("firm", "COMPETES_WITH", "firm"),   # competitive pressure
    ("firm", "CUSTOMERS_OF",  "firm"),   # downstream exposure
]
```

Key graph statistics derived for each node: betweenness centrality, degree centrality, PageRank, clustering coefficient.

### Section-Aware FinBERT Pipeline (Block B)

The critical architectural decision is processing MD&A and Risk Factor sections **independently** rather than concatenating raw text. Each section is chunked, encoded with frozen FinBERT, pooled via mean [CLS] aggregation, and then PCA-reduced to 64 dimensions separately. The two 64-d vectors are concatenated to form h_T ∈ ℝ¹²⁸.

This preserves the opposing economic signals that H2 tests: MD&A encodes strategic intent (positive correlation with CAR), Risk Factors encode liability exposure (negative correlation with CAR). Pooling them cancels both signals — the M2 Reversal.

### Leakage Prevention

```
Temporal partition:  [──── Train 2000–2016 ─────|░░|── Val 2017–2019 ──|░░|── Test 2020–2023 ──]
                                                 ↑↑                     ↑↑
                                          11-day embargo          11-day embargo
```

The 11-day embargo at each boundary excludes any deal whose ±5-day CAR event window overlaps the partition boundary, eliminating the Overlapping Outcomes leakage mechanism formalised by López de Prado (2018). All preprocessing (imputation, scaling, any trainable transformation) is fit on training folds only.

---

## The Deal Intelligence Terminal

The empirical findings are fully operationalised through an interactive **Streamlit research artefact** — not a static report, but a live system that renders every claim dynamically and allows deal-level interrogation.

### Terminal Pages

| Page | What it shows |
|------|--------------|
| **Model Evidence Wall** | Side-by-side M1 vs M3 ablation across AUC-ROC, F1, precision, recall with cross-validation error bars |
| **Deal-Level Diagnostics** | Per-deal synergy probability (calibrated M3 ensemble), betweenness centrality rank, MD&A semantic match score |
| **Topological Embeddings** (Zone 1) | Interactive acquirer–target supply-chain subgraph visualisation |
| **Semantic Radar** (Zone 2) | Pentagonal linguistic profile comparison between acquirer and target 10-K sections |
| **Glass Box / SHAP** (Zone 3) | SHAP waterfall decomposition with plain-English "Algorithmic Translation" |
| **Methodology Engine** | End-to-end M3 pipeline as interactive DAG with stage-specific output statistics |
| **Evaluation Lab** | Structured AUC-ROC defence with interactive ROC curve |
| **Hypothesis Lab** | Sector-wise AUC lift validating H1 with ±1σ error bars |
| **Pipeline Architecture** | Internal tensor flow and 249-dimensional concatenation logic |

---

## Project Structure

```
ma_project/
│
├── report/                         # Full dissertation (Typst source)
│   ├── main.typ                    # Root document
│   ├── chapters/                   # Ch1–Ch6 + appendices
│   └── references.bib              # BibTeX bibliography
│
├── frontend/                       # Deal Intelligence Terminal (Streamlit)
│   ├── 1_Deal_Terminal.py          # Entry point
│   ├── pages/
│   │   ├── 2_Model_Evidence_Wall.py
│   │   ├── 3_Deal_Diagnostics.py
│   │   ├── 4_Methodology_Engine.py
│   │   ├── 5_Evaluation_Lab.py
│   │   ├── 6_Hypothesis_Lab.py
│   │   └── 7_Pipeline_Architecture.py
│   └── components/                 # Reusable UI widgets
│
├── src/                            # Core library
│   ├── models/
│   │   ├── fusion.py               # Tri-modal late-fusion model (M3)
│   │   ├── graph_model.py          # HeteroGraphSAGE (Block C)
│   │   └── finbert_encoder.py      # Section-aware FinBERT (Block B)
│   ├── data/
│   │   ├── car_pipeline.py         # Two-stage CAR calculation
│   │   ├── temporal_split.py       # Embargo-enforced partitioning
│   │   └── edgar_scraper.py        # SEC EDGAR 10-K extraction
│   └── evaluation/
│       ├── ablation.py             # M1–M3e ablation runner
│       └── shap_analysis.py        # SHAP interpretation
│
├── scripts/                        # CLI entry points
│   ├── training/
│   │   ├── train.py                # Full training run
│   │   └── train_classifier.py     # Classifier pipeline only
│   ├── evaluation/
│   │   └── evaluate.py             # Held-out test set evaluation
│   └── figures/
│       └── generate_paper_figures.py   # All 6 supplementary plots
│
├── configs/                        # YAML experiment configurations
│   ├── full_fusion.yaml            # M3 configuration
│   ├── financial_only.yaml         # M1 baseline
│   └── ablation_suite.yaml         # M1–M3e sweep
│
├── data/
│   ├── raw/                        # Original downloads (gitignored)
│   ├── interim/                    # Cleaned, pre-feature matrices
│   └── processed/
│       └── final_car_dataset.csv   # Unified multimodal feature matrix
│
├── docs/
│   ├── figures/                    # All paper figures (PNG)
│   └── literature/                 # Reference PDFs
│
└── models/                         # Saved model checkpoints (gitignored)
    ├── m1_xgb_financial.pkl
    ├── m3_fusion_classifier.pkl
    └── graphsage_embeddings.pt
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+ required
# GPU optional but recommended for FinBERT encoding
python3 --version
```

### Installation

```bash
git clone https://github.com/HaardJoshi/ma_project.git
cd ma_project

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate          # macOS/Linux
# env\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Deal Intelligence Terminal

```bash
streamlit run frontend/1_Deal_Terminal.py
# Open http://localhost:8501
```

### Train Models

```bash
# Full M3 fusion model (requires processed data)
python scripts/training/train.py --config configs/full_fusion.yaml

# Financial-only baseline (M1)
python scripts/training/train.py --config configs/financial_only.yaml

# Run full ablation sweep (M1 → M3e)
python scripts/training/train.py --config configs/ablation_suite.yaml
```

### Reproduce Paper Figures

```bash
# Generates all 6 supplementary figures to docs/figures/
python scripts/figures/generate_paper_figures.py
```

### Compile Dissertation (requires Typst)

```bash
typst compile report/main.typ cn6000_final_submission.pdf
```

---

## Data Sources

| Source | Data Type | Coverage | Access |
|--------|-----------|----------|--------|
| **Yahoo Finance** | Financial ratios, equity returns | 2000–2023, US domestic M&A | Public API |
| **SEC EDGAR** | 10-K filings (MD&A + Risk Factors) | All US-listed acquirers/targets | Free (EDGAR API) |
| **Bloomberg SPLC** | Supply-chain relationship graph | ~3,000 firms | Institutional licence |
| **S&P 500 Index** | Market return benchmark (OLS market model) | Daily, 2000–2023 | Public |

> **Note:** Bloomberg SPLC data requires an institutional licence. The processed graph embeddings (non-recoverable to raw relationships) are included in `data/processed/` for reproducibility.

---

## Key Dependencies

```txt
torch>=2.0.0
torch-geometric>=2.4.0        # HeteroGraphSAGE
transformers>=4.35.0           # FinBERT
xgboost>=2.0.0
scikit-learn>=1.3.0
shap>=0.43.0
streamlit>=1.28.0
plotly>=5.18.0
pandas>=2.1.0
numpy>=1.26.0
statsmodels>=0.14.0            # OLS, Levene's test
```

---

## Research Context

### Performance Against the Literature

| Study | Method | Target | AUC / Acc. |
|-------|--------|--------|------------|
| Palepu (1986); Barnes (1990) | Logit / MDA | Acquisition likelihood | ~58% Acc. |
| Zhang et al. (2024) | XGBoost on ratios | Deal success proxies | 0.56–0.58* |
| Elhoseny et al. (2022) | Deep MLP | Financial distress | 0.958* |
| Hájek & Henriques (2024) | FinBERT sentiment | Acquisition occurrence | F1 0.71* |
| **This study — M1** | XGBoost (Financial) | Binary CAR direction | 0.5408 |
| **This study — M3** | Multimodal Fusion | Binary CAR direction | **0.5655** |

*Asterisked figures address different, structurally easier prediction targets. Direct comparison requires caution.*

### What This Study Contributes

1. **First multimodal M&A architecture combining** supply-chain GNNs + section-aware NLP + financial fundamentals directed at binary CAR direction classification
2. **The M2 Reversal** — empirical proof that naive NLP actively destroys predictive value in M&A models unless filing-section semantics are preserved; a methodological warning for future researchers
3. **A boundary condition** — negative R² across all regression configurations precisely defines where multimodal architecture can and cannot help in M&A prediction
4. **The Deal Intelligence Terminal** — a reproducible, interactive artefact rendering all empirical claims dynamically

---

## Academic Details

```
Title:      Breaking the Tabular Ceiling: Heterogeneous Graph-Aware
            Multimodal Fusion for M&A Synergy Prediction
Author:     Hard Joshi (Student ID: 2512658)
Degree:     BSc (Hons) Data Science and Artificial Intelligence
Module:     CN6000 — Final Year Project
Supervisor: Arish Siddiqui
University: University of East London
Submitted:  08 May 2026
```

---

## Licence

This repository is released under the [MIT Licence](LICENSE). The dissertation text (`report/`) is © Hard Joshi 2026 — all rights reserved. Bloomberg SPLC data is excluded from the repository under institutional data licence terms.

---

<div align="center">

*"The magnitude of short-window announcement returns remains dominated by unobservable and idiosyncratic shocks. Multimodal architecture helps more with sign discrimination than with point-estimation of return magnitude — and knowing that boundary is itself a contribution."*

</div>
