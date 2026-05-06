// ============================================================
//  appendix-b.typ
//  Appendix B: Final Project Proposal
//  M&A Synergy Prediction | Hard Joshi | UEL
// ============================================================

= Final Project Proposal <appendix-proposal>

This appendix reproduces the final project proposal as submitted for CN6000 approval. The document has been formatted for inclusion in the dissertation but the substantive content is unchanged from the approved submission.

== Project Metadata

#figure(
  table(
    columns: (1fr, 3fr),
    align: (left, left),
    inset: 8pt,
    stroke: 0.5pt,
    [*Module*], [CN6000 --- Final Year Project],
    [*Student*], [Hard Joshi (2512658)],
    [*Supervisor*], [Arish Siddiqui],
    [*Programme*], [BSc Hons Data Science and Artificial Intelligence],
    [*University*], [University of East London],
  ),
  caption: [Project metadata as submitted for CN6000 approval.],
)

== Project Title

_Breaking the Tabular Ceiling: Heterogeneous Graph-Aware Multimodal Fusion for M&A Synergy Prediction_

== Revised Aim

The aim of this project is to construct, evaluate, and interpret a heterogeneous multimodal machine learning framework for predicting M&A synergy outcomes, fusing financial fundamentals, section-conditioned regulatory filing semantics, and inter-firm supply-chain graph topology into a unified prediction architecture. The central thesis is that every prior generation of quantitative M&A predictor fails not because of insufficient computational power, but because of a flawed representational assumption: treating each firm as a topologically isolated data point.

== Research Hypotheses

The project is structured around three falsifiable hypotheses, each corresponding to a distinct information modality:

*H1 --- Topological Alpha Hypothesis:*
The inclusion of heterogeneous supply-chain graph topology (Block C, HeteroGraphSAGE embeddings) will yield a statistically significant improvement in directional deal discrimination (AUC-ROC) over a financial-only baseline, proving that topological data yields orthogonal predictive alpha that is irreducible from financial ratios.

*H2 --- Semantic Divergence Hypothesis:*
The relationship between textual similarity and announcement returns is conditional on document section. MDA similarity (strategic fit) correlates positively with CAR; Risk Factor similarity (shared liability exposure) correlates negatively. These opposing effects are mathematically suppressed when sections are pooled into a single embedding, rendering naive FinBERT applications harmful.

*H3 --- Topological Arbitrage / Information Transparency Dampening Hypothesis:*
Acquirers with high betweenness centrality in the supply-chain graph exhibit statistically compressed variance in announcement return outcomes relative to peripheral acquirers, as measured by Levene's test across centrality quantile groups, consistent with the information transparency hypothesis.

== Scope and Dataset

=== Deal Universe

The study is scoped to completed US domestic M&A transactions announced between 2000 and 2023, comprising approximately 3,000 deals with full multimodal coverage. The deal universe is sourced from Yahoo Finance and LSEG Refinitiv, filtered to transactions where both acquirer and target have full financial fundamentals, pre-announcement 10-K filing history on SEC EDGAR, and supply-chain node presence in the Bloomberg SPLC dataset.

=== Feature Blocks

Three complementary information modalities are integrated:

#figure(
  table(
    columns: (1fr, 2fr, 2fr, 1fr),
    align: (center, left, left, center),
    inset: 8pt,
    stroke: 0.5pt,
    table.header(
      [*Block*], [*Modality*], [*Source*], [*Dimension*],
    ),
    [A], [Financial Fundamentals], [Yahoo Finance / LSEG Refinitiv], [56 ratios],
    [B], [Section-Aware Textual Semantics], [SEC EDGAR (10-K: MDA + Risk Factors)], [FinBERT pairwise cosine],
    [C], [Heterogeneous Graph Topology], [Bloomberg SPLC], [64-dim embeddings],
  ),
  caption: [Summary of multimodal feature blocks.],
)

=== Target Variable

The prediction target is binary Cumulative Abnormal Return (CAR) direction over an 11-day event window ($t minus 5$ to $t plus 5$ relative to announcement date), computed against an OLS market model estimated on a 252-day pre-event benchmark window. The binary encoding (1 = CAR > 0, value-creating; 0 otherwise) is chosen due to the well-documented noise characteristics of short-window CAR magnitudes, which are structurally resistant to regression modelling.

=== Leakage Prevention

To prevent look-ahead contamination and market microstructure leakage, the pipeline enforces:
- Strict temporal train/test splits (no fold contains future deals relative to the test period);
- An 11-day event-window embargo on all samples whose event windows overlap with fold boundaries;
- Explicit exclusion of acquisition-relationship edges from the supply-chain graph during training.

== Architecture

=== Feature Extraction

- *Block A:* Financial ratios covering leverage, liquidity, profitability, deal structure, and size metrics for both acquirer and target. Missing values imputed via median strategy.
- *Block B:* Section-specific FinBERT embeddings extracted independently from MDA and Risk Factor sections of the most recent pre-announcement 10-K filing. Acquirer-target pairwise cosine similarity scores computed separately per section, preserving their opposing economic directionality.
- *Block C:* Two-hop heterogeneous GraphSAGE trained on Bloomberg SPLC supply-chain edges, distinguishing supplier, customer, and competitor relationship types via heterogeneous edge-type encoding. Each firm node is represented as a 64-dimensional embedding vector.

=== Fusion Strategy

Late fusion via concatenation: each modality is independently encoded into a fixed-dimensional embedding, and the three vectors are concatenated into a joint representation before a shared XGBoost prediction head. This design preserves modality-specific representation learning, is robust to incomplete modality coverage, and enables clean SHAP attribution per modality block.

=== Evaluation Framework

A dual-pipeline evaluation is applied:
- *Classification pipeline:* Predicts binary CAR direction; evaluated on AUC-ROC (primary), accuracy, F1, and precision-recall curves.
- *Regression pipeline:* Predicts continuous CAR magnitude; evaluated on $R^2$ and MAE.

Both pipelines are evaluated under five-fold stratified cross-validation with the temporal split and event-window embargo described above.

=== Hypothesis Testing Protocol

#figure(
  table(
    columns: (2fr, 2fr, 1.5fr),
    align: (left, left, left),
    inset: 8pt,
    stroke: 0.5pt,
    table.header(
      [*Hypothesis*], [*Test*], [*Correction*],
    ),
    [H1: Topological Alpha], [Paired t-test on fold-level AUC (M1 vs M3)], [Bonferroni],
    [H2: Semantic Divergence], [OLS regression on semantic-divergence subsample], [Bonferroni],
    [H3: Topological Arbitrage], [Levene's test across centrality quantile groups], [Bonferroni],
  ),
  caption: [Formal hypothesis testing protocol.],
)

== Ablation Ladder

To isolate the marginal contribution of each modality block, the following model configurations are evaluated:

#figure(
  table(
    columns: (1fr, 2fr, 2fr),
    align: (center, center, left),
    inset: 8pt,
    stroke: 0.5pt,
    table.header(
      [*Model*], [*Blocks Included*], [*Purpose*],
    ),
    [M1], [A only], [Financial baseline],
    [M2], [A + B (pooled)], [Naive NLP augmentation],
    [M2s], [A + B (section-split)], [Section-aware NLP],
    [M3], [A + B (section-split) + C], [Full multimodal],
    [M3e], [A + scalar centrality + C], [GNN vs. scalar ablation],
  ),
  caption: [Ablation ladder configuration.],
)

== Research Output Artefact

All empirical results are operationalised through the *Deal Intelligence Terminal*, an interactive full-stack research artefact built in Streamlit/Python. The terminal renders the ablation results, hypothesis test statistics, SHAP feature attributions, and deal-level prediction scores dynamically, ensuring full reproducibility and providing a production-grade demonstration of the model's practical utility.

== Key Divergences from Initial Proposal

#figure(
  table(
    columns: (2fr, 2fr, 2fr),
    align: (left, left, left),
    inset: 8pt,
    stroke: 0.5pt,
    table.header(
      [*Initial Proposal*], [*Final Implementation*], [*Reason for Change*],
    ),
    [Integrate financial data + sentiment from news articles], [Replaced news with pre-announcement 10-K filings], [Filings are temporally precise, audited, and free from bias],
    [Generic FinBERT sentiment], [Section-split MDA vs. Risk Factor similarity], [Discovered pooling cancels opposing signals (M2 reversal)],
    [Crunchbase as data source], [Replaced with Yahoo Finance + Bloomberg SPLC + SEC EDGAR], [Crunchbase lacks coverage for pre-2015 institutional deals],
    [ML framework for prediction accuracy], [Extended to formal hypothesis testing + GNN architecture], [Literature review revealed architectural limitations],
    [Predict M&A success], [Predict binary CAR direction], [CAR is less behaviourally contaminated than accounting measures],
  ),
  caption: [Summary of project trajectory and methodological divergences.],
)

== Ethical, Legal, and Social Considerations

All datasets used are publicly available or accessed via institutional licences (Bloomberg Terminal, LSEG Refinitiv). No human participants are involved. SEC filings are public regulatory documents. Supply-chain data is aggregated and does not include proprietary commercial relationships. Model outputs are not used for live trading or commercial decision-making within the scope of this study.

#v(2em)
#align(center)[
  _Submitted in partial fulfilment of CN6000, BSc Hons Data Science and Artificial Intelligence, University of East London._
]
