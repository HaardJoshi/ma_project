# M&A Synergy Prediction: Multimodal Deal Intelligence Terminal

A publication-ready research framework and interactive terminal for predicting post-acquisition synergy (Cumulative Abnormal Return). The system uses a Multimodal Heterogeneous Graph Neural Network (HGNN) to fuse financial metrics (Block A), textual strategic intent via FinBERT (Block B), and topological ecosystem data via GraphSAGE (Block C).

## Research Outcomes

This project successfully established the following empirical anchors:
- **Topological Alpha (H1):** The inclusion of supply-chain graph topology yielded a statistically significant improvement in directional deal discrimination (AUC-ROC) over financial-only baselines.
- **Semantic Divergence (H2):** The discovery of the "M2 Reversal" - showing that undifferentiated text aggregation hides predictive signal, while section-specific processing (MD&A vs. Risk Factors) recovers it.
- **Topological Arbitrage (H3):** Evidence that firm centrality in industrial networks compresses the variance of announcement-period returns, validated via Levene's test.

## Quick Start

```bash
# 1. Clone & setup
git clone <repo-url> && cd ma_project
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt

# 2. Run the Deal Intelligence Terminal (Dashboard)
streamlit run frontend/1_Deal_Terminal.py

# 3. Train models
python scripts/training/train.py --config configs/full_fusion.yaml

# 4. Compile Report (Requires Typst)
typst compile report/main.typ dissertation_final.pdf
```

## Project Structure

```
├── report/                     Final dissertation (Typst source)
├── frontend/                   Streamlit "Deal Intelligence Terminal"
├── src/                        Core library (Modelling & Data Pipelines)
├── scripts/                    CLI entry points for Training & Evaluation
├── configs/                    YAML experiment configurations
├── data/                       Interim and processed feature matrices
├── docs/                       Figures, assets, and literature review
└── models/                     Saved model checkpoints
```

## Architecture

The model follows a 3-block late-fusion architecture:

| Block | Input | Module | Output |
|-------|-------|--------|--------|
| **A - Financial** | 50+ financial ratios (LSEG) | Standardisation -> Ridge/MLP | h_F |
| **B - Text** | 10-K MD&A + Risk Factors (EDGAR) | Frozen FinBERT -> [CLS] | h_T |
| **C - Graph** | Supply-chain/competitor network (SPLC) | Degree/centrality + GraphSAGE | h_G |

**Fusion**: `z_i = [h_F || h_T || h_G]` -> MLP prediction head -> predicted CAR direction.

## Final Note
This repository contains the complete codebase and dissertation report for the BSc Computer Science final project. It represents the culmination of a research programme into multimodal financial machine learning and industrial topology.

*Good fun.*
