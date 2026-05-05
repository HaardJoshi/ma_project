// ============================================================
//  00-abstract.typ  (v1 — Final)
//  Abstract — 498 words
//  M&A Synergy Prediction | Hard Joshi | UEL
// ============================================================
#set par(justify: true, leading: 0.65em)

= Abstract <ch-abstract>

Mergers and acquisitions destroy shareholder value in 70–90% of cases, yet
existing predictive models consistently fail to identify value-destroying deals
before announcement. This study argues that the root cause is architectural
rather than computational: every prior generation of quantitative M&A model
— from logistic regression to transformer-based NLP — treats each firm as an
isolated data point, blind to the supply-chain topology, competitive network
structure, and section-level semantic signals that collectively determine
whether a proposed combination will create or destroy value. This dissertation
addresses that gap by constructing, evaluating, and interpreting a
heterogeneous multimodal architecture that encodes all three information
dimensions simultaneously, and by formally testing whether each dimension
carries predictive signal that is irreducible from the others.

A tri-modal late-fusion model (HeteroGraphSAGE) is built and evaluated on a
universe of completed US domestic M&A transactions announced between 2000 and
2023, comprising approximately 3,000 deals with full multimodal coverage,
sourced from LSEG Refinitiv, SEC EDGAR, and Bloomberg SPLC. Three feature
blocks are fused via late concatenation: 56 financial ratio features (Block A),
section-conditioned FinBERT embeddings of Management Discussion & Analysis and
Risk Factor disclosures extracted separately from pre-announcement 10-K filings
(Block B), and two-hop heterogeneous GraphSAGE embeddings derived from
firm-level supply-chain networks (Block C). A dual-evaluation framework is
applied: a classifier pipeline that predicts binary Cumulative Abnormal Return
(CAR) direction and a regression pipeline that predicts CAR magnitude. Both
pipelines are evaluated under five-fold stratified cross-validation with strict
temporal splits and an 11-day event-window embargo to prevent forward-looking
leakage. Three formal hypotheses are tested — H1 (Topological Alpha), H2
(Semantic Divergence), and H3 (Topological Arbitrage) — using paired t-tests,
OLS regression, and Levene's variance test respectively, with Bonferroni
correction applied across all three tests.

All three hypotheses are supported. The full multimodal model achieves
AUC-ROC = 0.5655, a statistically meaningful +0.0247 lift over the
financial-only baseline (0.5408), confirming that supply-chain topology
encodes directional signal that tabular models cannot recover (H1). OLS
estimation on a semantic-divergence subsample of 1,140 deals recovers
opposite-signed coefficients for MD&A similarity ($beta = +0.0044$) and
Risk Factor similarity ($beta = -0.0080$), confirming that section-level
semantic divergence is economically directional when the two sections are
modelled separately rather than pooled (H2). Levene's test across
betweenness-centrality quantile groups yields $F = 7.07$ ($p = 0.0079$),
confirming that structurally central acquirers experience statistically
compressed announcement-return variance (H3). The study additionally
establishes that adding undifferentiated FinBERT text to the financial
baseline reduces AUC by $-0.012$, demonstrating that naive NLP actively
destroys predictive value when filing-section semantics are conflated — a
methodological finding with direct implications for subsequent M&A NLP
research. Continuous CAR magnitude remains intractable to regression across
all configurations ($R^2 < 0$), precisely locating the boundary between
tractable and structurally difficult M&A prediction sub-problems. All
empirical results are rendered dynamically through the Deal Intelligence
Terminal, an interactive research artefact ensuring full reproducibility.
