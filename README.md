# M&A Synergy Prediction

Multimodal Heterogeneous Graph Neural Network (HGNN) framework for predicting post-acquisition synergy (Cumulative Abnormal Return) by fusing financial metrics, textual strategic intent (FinBERT), and topological ecosystem data (GraphSAGE).

## Quick Start

```bash
# 1. Clone & setup
git clone <repo-url> && cd ma_project
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt

# 2. Data pipeline (if starting from raw exports)
make combine       # merge 5 LSEG CSV exports → data/interim/ma_combined.csv
make clean-data    # clean → data/interim/ma_cleaned.csv
make preprocess    # winsorise, z-score, split → data/processed/

# 3. Train
python scripts/training/train.py --config configs/financial_only.yaml   # Block A baseline
python scripts/training/train.py --config configs/full_fusion.yaml      # Full multimodal

# 4. Evaluate
python scripts/training/evaluate.py --config configs/financial_only.yaml
```

## Project Structure

```
├── configs/                    YAML experiment configurations
├── src/                        Core library (importable package)
│   ├── config.py               Config loader & path resolver
│   ├── data/                   Data loading, cleaning, preprocessing
│   ├── features/               Feature extraction (Blocks A, B, C)
│   ├── models/                 Prediction models (baselines, MLP, fusion)
│   ├── training/               Training loop, CV, checkpointing
│   └── evaluation/             Metrics, results export
├── scripts/                    CLI entry points & pipeline scripts
│   ├── data/                   Data ingestion, cleaning, merging, CAR
│   ├── features/               Bloomberg, SPLC, EDGAR, text extraction
│   ├── graphs/                 Graph construction & GNN training
│   ├── training/               Model training, tuning, entry points
│   └── evaluation/             Hypothesis tests & figure generation
├── frontend/                   Streamlit dashboard application
│   ├── 1_Deal_Terminal.py      Main deal diagnostic page
│   └── pages/                  Additional dashboard pages
├── data/                       Datasets (git-ignored)
│   ├── raw/                    Original LSEG CSV exports
│   ├── interim/                Combined + cleaned CSVs
│   └── processed/              Final feature matrices + JSON caches
├── models/                     Saved checkpoints (git-ignored)
├── results/                    Evaluation CSVs & hypothesis outputs
├── docs/                       Dissertation & reference documents
├── notebooks/                  Exploratory analysis
├── lib/                        Vendored JS libraries (vis.js, etc.)
└── diagram_examples/           Reference architecture diagrams
```

## Architecture

The model follows a 3-block architecture:

| Block | Input | Module | Output |
|-------|-------|--------|--------|
| **A — Financial** | 50+ financial ratios (LSEG) | Standardisation → Ridge/MLP | h_F |
| **B — Text** | 10-K MD&A + Risk Factors (EDGAR) | Frozen FinBERT → [CLS] | h_T |
| **C — Graph** | Supply-chain/competitor network (SPLC) | Degree/centrality + GraphSAGE | h_G |

**Fusion**: `z_i = [h_F ∥ h_T ∥ h_G]` → MLP prediction head → predicted CAR

## Experiments

| Config | Features | Hypothesis |
|--------|----------|------------|
| `financial_only.yaml` | F only | Baseline |
| `financial_text.yaml` | F + T | H2: Semantic Divergence |
| `financial_graph.yaml` | F + G | H1: Topological Alpha |
| `full_fusion.yaml` | F + T + G | H3: Topological Arbitrage |

## Cloud GPU

```bash
# On any cloud instance with GPU:
git clone <repo-url> && cd ma_project
pip install -r requirements.txt

# Upload data to data/interim/ (or run make combine && make clean-data)
# Then:
python scripts/training/train.py --config configs/full_fusion.yaml
```

The config auto-detects CUDA / MPS / CPU — no code changes needed.
