# Revised Literature Review — Changelog

This document tracks every change made from the original `02-lit-review.typ` to the revised version.

## Structural Changes

| # | Type | Section | Change |
|---|------|---------|--------|
| 1 | **NEW** | §2.1.4 | Added "Market-Driven Acquisitions" subsection (Shleifer & Vishny, 2003) |
| 2 | **NEW** | §2.3.4 | Added "The Case for Directional Classification" — justifies regression→classification pivot with critical evaluation of AUC-ROC vs Accuracy vs F1 |
| 3 | **REWRITTEN** | §2.4.2 | Renamed to "From Transductive to Inductive Learning"; added GCN context (Kipf & Welling, 2017) before GraphSAGE; added justification for choosing GraphSAGE over GCN/GAT/GIN |
| 4 | **REWRITTEN** | §2.5.1 | Renamed to "The Case for Heterogeneous Late Fusion"; added late-fusion vs end-to-end justification; added XGBoost selection rationale over RF/LightGBM; added SHAP justification (Lundberg & Lee, 2017) |
| 5 | **REWRITTEN** | §2.5.2 H₁ | R² → AUC-ROC; added paired t-test justification |
| 6 | **REWRITTEN** | §2.5.2 H₃ | Removed "GNN attention weights"; replaced with Levene's test on betweenness quantiles; added Levene vs Bartlett justification |
| 7 | **EXPANDED** | §2.5.2 H₂ | Added critical evaluation of why bivariate OLS is used over non-linear alternatives |

## Citation Fixes

| # | Action | Citation | Reason |
|---|--------|----------|--------|
| 1 | **REMOVED** | `@Anderson_2018` | Was a Pew Research teen social media report — completely unrelated |
| 2 | **REPLACED WITH** | `@betton_eckbo_thorburn_2008` | Comprehensive handbook on corporate takeovers documenting low R² |
| 3 | **ADDED** | `@shleifer_vishny_2003` | Market-timing M&A theory |
| 4 | **ADDED** | `@kipf_welling_2017` | GCN foundation paper |
| 5 | **ADDED** | `@lundberg_shap_2017` | SHAP explainability framework |
| 6 | **ADDED** | `@ajayi_ml_ma_2022` | ML for M&A binary classification |
| 7 | **NOW CITED IN LIT REVIEW** | `@3.1Baltrušaitis-Ahuja` | Was only in methodology bib; now referenced in §2.5.1 |
| 8 | **NOW CITED IN LIT REVIEW** | `@3.1Chen_2016` | Was only in methodology bib; now referenced in §2.5.1 |

## Typo & Language Fixes

| Line (original) | Before | After |
|-----------------|--------|-------|
| L9 | `The. Behavioral` | `The Behavioural` |
| L11 | `super addictive` | `super-additive` |
| L27 | `dependancies` | `dependencies` |
| L113 | `Heterogenous` | `Heterogeneous` |
| L115 | `complimentary` | `complementary` |
| Various | US spelling | Standardised to UK spelling (behaviour, analyse, utilise, etc.) |

## Critical Evaluations Added

The revised version includes explicit "why this method over alternatives" justifications for:

1. **AUC-ROC over Accuracy/F1** — threshold-invariance under class imbalance
2. **Binary classification over regression** — noise floor of continuous CAR
3. **XGBoost over RF/LightGBM/MLP** — regularisation, missing-value handling, high feature-to-sample ratio
4. **Late fusion over end-to-end** — sample size constraints (N≈2,800)
5. **GraphSAGE over GCN/GAT/GIN** — inductive capability, heterogeneous edges, mini-batch scalability
6. **Paired t-test for H₁** — naturally paired CV folds
7. **OLS for H₂** — interpretable coefficient signs for directional hypothesis
8. **Levene's over Bartlett's for H₃** — non-normality of |CAR|, unequal group sizes
9. **SHAP over permutation importance/attention weights** — game-theoretic foundation, correlation robustness
