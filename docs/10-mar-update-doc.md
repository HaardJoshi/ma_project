# Dissertation Implementation Progress Update

**Date:** 10 March 2026  
**To:** Dr. Arish Siddiqui  
**From:** Hard Joshi  
**Subject:** M&A Synergy Prediction — Implementation Progress & Next Steps

---

## ✅ Completed

| # | Task | Notes |
|---|---|---|
| 1 | **Data collection & cleaning** | 4,999 M&A deals from LSEG (1995–2022), merged from 5 CSV exports, cleaned and de-duplicated |
| 2 | **Project structure & reproducibility** | Modular Python package, YAML configs, Makefile, Git version control on GitHub |
| 3 | **CAR computation (target variable)** | Market model OLS (estimation window -200 to -20, event window -5 to +5). 4,509/4,999 deals (90.2%) have valid 11-day CAR. Mean acquirer CAR = -0.88% |
| 4 | **SEC EDGAR 10-K pipeline** | Automated download of 2,056 filings. Extracted Item 7 (MD&A) and Item 1A (Risk Factors) sections |
| 5 | **Block B: FinBERT text embeddings** | Frozen ProsusAI/finbert, penultimate layer [CLS] extraction, chunked for long documents, PCA 768→64 dims (96.5% variance retained). 1,674 deals with text features |
| 6 | **Combined dataset** | `final_car_dataset.csv` — 4,999 rows × 203 columns (67 financial + 128 text + CAR metrics) |

## 🔄 In Progress

| # | Task | ETA |
|---|---|---|
| 7 | **Block C: Supply chain graph data** | ~1 week — Bloomberg SPLC pull script ready, need to execute on Terminal |

## 📋 Next Steps

| # | Task | ETA |
|---|---|---|
| 8 | Baseline models (financial-only): Ridge, ElasticNet, XGBoost → predict CAR | ~2-3 days |
| 9 | Financial + Text model — test if NLP features add predictive power | ~1-2 days |
| 10 | Graph feature extraction (GraphSAGE / network metrics) | ~1 week |
| 11 | Full fusion model (all three blocks) | ~1 week |
| 12 | Hypothesis testing, ablation studies, evaluation | ~1 week |
| 13 | Results integration into dissertation chapters | Ongoing |

---

## Key Metrics

- **4,999** M&A deals collected
- **4,509** with valid CAR (90.2% coverage)
- **1,674** with FinBERT text embeddings (33.5% — US acquirers with EDGAR filings)
- **96.5%** PCA variance retained at 64 dimensions
- Mean acquirer CAR: **-0.88%** (consistent with academic literature)

---

## Publishability — Novel Contributions

I believe this work has several elements of novelty that could make it suitable for publication. I would greatly appreciate your assessment:

### 1. Novel Multimodal Dataset
To the best of my knowledge, no existing public dataset combines **financial deal characteristics + 10-K textual embeddings + supply chain graph topology** for M&A synergy prediction. I am constructing this dataset from scratch by integrating:
- LSEG deal-level financial data (Data fetched from Bloomberg Terminal)
- SEC EDGAR 10-K filings (automated extraction pipeline)
- Bloomberg supply chain relationships (SPLC)
- Event study CARs computed from Yahoo Finance + Bloomberg price data

This dataset itself could be a contribution to the field.

### 2. FinBERT for M&A Prediction (Underexplored)
While FinBERT has been applied to sentiment analysis and credit risk, its application to **M&A synergy prediction using 10-K filing sections** (specifically MD&A and Risk Factors) is relatively novel. Most M&A prediction literature still relies on financial ratios alone or uses basic bag-of-words text representations.

### 3. Supply Chain Graphs in M&A (Novel)
Incorporating **supply chain network topology** (via Graph Neural Networks / GraphSAGE) into M&A outcome prediction is, to my knowledge, largely unexplored. The hypothesis — that acquirer-target supply chain overlap or complementarity predicts synergy — has theoretical backing but limited empirical testing with modern graph ML methods.

### 4. Multimodal Fusion Architecture
The three-block fusion approach (Financial → Text → Graph) allows systematic **ablation testing**: we can rigorously measure the marginal contribution of each modality. This addresses a gap in the literature where multimodal financial models are often presented without proper ablation.

### 5. Reproducible Pipeline
The entire pipeline — from data collection to model evaluation — is version-controlled, config-driven, and reproducible. This addresses growing concerns about reproducibility in financial ML research.

---

## Questions for Your Feedback

1. Is the 33.5% text coverage (US acquirers with EDGAR filings) sufficient, or should I investigate expanding it?
2. Should I prioritise baseline models before or after completing the SPLC graph data?
3. Any concerns about the methodology so far (CAR computation, FinBERT approach)?
4. **Do you think this work is publishable?** If so, which venue would you recommend targeting — a finance journal, an AI/ML conference, or an interdisciplinary outlet?
5. Are there any additional novelty angles I should emphasise or develop further?

Best regards,  
Hard Joshi
