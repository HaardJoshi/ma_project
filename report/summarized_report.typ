// ============================================================
//  summarized_report.typ
//  Supervisor Summary — Hard Joshi (Student ID: 2512658)
//  M&A Synergy Prediction via HeteroGraphSAGE
//  Data Science and Artificial Intelligence — UEL CN6000
// ============================================================

#set document(author: "Hard Joshi", title: "Supervisor Summary — M&A Synergy Prediction")
#set page(
  paper: "a4",
  margin: (left: 2.5cm, right: 2.5cm, top: 2.5cm, bottom: 2.5cm),
  numbering: "1",
  number-align: center,
)
#set text(font: "Times New Roman", lang: "en", size: 11pt)
#set par(justify: true, leading: 0.7em)
#show heading.where(level: 1): it => {
  v(1.2em)
  text(size: 13pt, weight: "bold")[#it.body]
  v(0.5em)
}
#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 11pt, weight: "bold", style: "italic")[#it.body]
  v(0.3em)
}

// ── HEADER ─────────────────────────────────────────────────
#align(center)[
  #text(size: 15pt, weight: "bold")[Supervisor Summary Report]
  #v(0.4em)
  #text(size: 12pt)[
    _AI-Driven M&A Synergy Prediction via Heterogeneous Graph Neural Networks_
  ]
  #v(0.6em)
  #grid(
    columns: (1fr, 1fr),
    row-gutter: 0.3em,
    align(left)[*Student:* Hard Joshi],       align(right)[*ID:* 2512658],
    align(left)[*Degree:* Data Science & AI], align(right)[*Module:* CN6000],
    align(left)[*Supervisor:* Arish Siddiqui], align(right)[*Date:* 8 May 2026],
  )
]

#line(length: 100%, stroke: 1pt)
#v(0.5em)

// ── 1. PROBLEM STATEMENT ───────────────────────────────────
= Problem Statement

M&A transactions destroy shareholder value in 70–90% of cases. Despite this, existing predictive models consistently fail to identify value-destroying deals before announcement. This study argues the failure is architectural: every prior quantitative M&A model treats firms as isolated balance-sheet vectors, blind to the supply-chain networks, competitive topologies, and section-level semantic signals that actually determine whether a deal will create or destroy value.

// ── 2. RESEARCH DESIGN ─────────────────────────────────────
= Research Design

A tri-modal late-fusion architecture (*HeteroGraphSAGE*) was constructed and evaluated on ~3,000 completed US domestic M&A transactions (2000–2023), sourced from Yahoo Finance, SEC EDGAR, and Bloomberg SPLC.

#v(0.3em)
#table(
  columns: (1.5cm, 2.5cm, auto),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { luma(235) },
  table.header([*Block*], [*Modality*], [*Description*]),
  [A], [Financial],   [56 ratio features: leverage, liquidity, profitability, deal structure],
  [B], [Textual],     [FinBERT embeddings of MD&A and Risk Factors from pre-deal 10-K filings, kept section-separate to preserve opposing economic signals],
  [C], [Graph],       [Two-hop heterogeneous GraphSAGE embeddings from Bloomberg SPLC supply-chain network],
)
#v(0.3em)

Three formal hypotheses were tested against Bonferroni-corrected thresholds ($alpha = 0.0167$):
- *H1 — Topological Alpha:* Graph topology adds predictive signal beyond financial features.
- *H2 — Semantic Divergence:* MD&A and Risk Factor sections encode economically opposing signals.
- *H3 — Topological Arbitrage:* Network centrality compresses announcement-return variance.

Evaluation used five-fold stratified cross-validation with strict temporal splits (train: 2000–2016 / val: 2017–2019 / test: 2020–2023) and an 11-day event-window embargo to prevent leakage.

// ── 3. KEY RESULTS ─────────────────────────────────────────
= Key Empirical Results

== Classification Pipeline (AUC-ROC — Direction Prediction)

#table(
  columns: (2cm, 3cm, 2cm, 2cm, 2cm),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { luma(235) },
  table.header([*Model*], [*Configuration*], [*AUC*], [*Acc.*], [*F1*]),
  [M1], [Financial only],    [0.5408], [52.8%], [0.473],
  [M2], [Financial + Text],  [0.5289], [52.9%], [0.476],
  [*M3*], [*Full Fusion*],   [*0.5655*], [*54.8%*], [*0.490*],
  [M3e], [M3 + Aux Feats.],  [0.5585], [55.1%], [0.492],
)

#v(0.3em)
The headline finding: *M3 achieves AUC = 0.5655, a +0.0247 lift over the financial-only baseline.* Hyperparameter tuning alone did not improve any model, confirming that architectural signal choice matters more than optimiser search.

#v(0.3em)
A key negative result: M2 (Financial + naive text) *reduces* AUC by −0.0119 versus M1. This is the M2 Reversal — adding undifferentiated FinBERT text actively destroys predictive value when MD&A and Risk Factor signals are pooled without section-separation.

== Regression Pipeline (Continuous CAR Magnitude)

All regression configurations produced $R^2 < 0$, meaning the model cannot explain continuous CAR magnitude better than the sample mean. This is an honest and informative negative result: short-window announcement returns are dominated by idiosyncratic shocks (competing bids, investor sentiment, payment structure) that are structurally invisible in pre-announcement data. The boundary between tractable (direction) and intractable (magnitude) prediction sub-problems is precisely located.

== Hypothesis Test Verdicts

#table(
  columns: (1cm, 3cm, 3.5cm, 3cm),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { luma(235) },
  table.header([*H*], [*Claim*], [*Evidence*], [*Verdict*]),
  [H1], [Topology adds signal],
        [AUC lift +0.0247; paired $t$-test, $p < 0.0167$],
        [✓ *Supported*],
  [H2], [Section semantics diverge],
        [$beta_"MDA" = +0.0044$, $beta_"RF" = -0.0080$; OLS $p < 0.0167$],
        [✓ *Supported*],
  [H3], [Centrality dampens variance],
        [$F_"Levene" = 7.07$, $p = 0.0079$; $r = -0.0701$, $p = 0.0002$],
        [✓ *Supported*],
)

// ── 4. CONTRIBUTIONS ───────────────────────────────────────
= Research Contributions

+ *Methodological:* One of the first explicit applications of heterogeneous GNNs to post-merger CAR classification, formally moving the field beyond the tabular independence assumption.
+ *Empirical:* Establishes that section-level semantic divergence between MD&A and Risk Factor disclosures is directional and economically meaningful in M&A outcome prediction. Conflating these sections is demonstrably harmful.
+ *Practical Artefact:* The *Deal Intelligence Terminal* — a fully interactive research dashboard rendering all model outputs, ablation ladders, SHAP decompositions, and hypothesis tests dynamically — ensures complete reproducibility.

// ── 5. PRACTICAL SIGNIFICANCE ──────────────────────────────
= Practical Significance of +0.025 AUC

A +0.025 AUC improvement means the model is 2.5 percentage points more likely than chance to correctly rank a value-creating deal above a value-destroying one when comparing random deal pairs. In advisory practice, M&A evaluation is a ranking problem: teams need a better triage of which opportunities deserve deeper diligence, not perfect foresight. Applied across a large deal pipeline, this persistent edge translates into systematically better capital allocation.

// ── 6. LIMITATIONS ─────────────────────────────────────────
= Key Limitations

- The headline AUC of 0.5655 is meaningful for research but not commercially deployable as a standalone decision system.
- The text pipeline uses PCA compression of section embeddings; a cross-attention dual-stream encoder would likely recover additional semantic signal.
- H3 is a structural finding: high-centrality firms may also be larger and more analyst-covered, so centrality cannot be claimed as an exhaustively isolated causal driver.
- The sample is US-listed firms only; generalisability to cross-border or private-equity transactions remains untested.

// ── 7. ASSESSMENT COMPLIANCE ───────────────────────────────
= Handbook Compliance Summary (CN6000)

#table(
  columns: (auto, 1.5cm),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { luma(235) },
  table.header([*Criterion*], [*Met*]),
  [Significant non-trivial computing problem solved],         [✓],
  [Literature review analysing prior work],                   [✓],
  [Specification, design, and implementation of a system],    [✓],
  [Ethical, legal, and social considerations addressed],      [✓],
  [Evaluation of own project work against research of others],[✓],
  [Harvard (Cite Them Right) referencing],                    [✓],
  [Ethical approval submitted],                               [✓],
  [Personal reflection and Mental Wealth section],            [✓],
  [Interactive artefact (Deal Intelligence Terminal)],        [✓],
  [~10,000 word report],                                      [✓],
)

#v(1em)
#line(length: 100%, stroke: 0.5pt)
#v(0.3em)
#align(center)[
  #text(size: 9pt, style: "italic")[
    This summary was prepared alongside the full dissertation submitted for CN6000 on 8 May 2026. \
    Full report: #text(weight: "bold")[cn6000_final_submission.pdf]
  ]
]
