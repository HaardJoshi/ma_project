// ============================================================
// Chapter 3: Methodology  — Polished with Architecture Diagram + Tables
// Hard Joshi | BSc Data Science & AI | University of East London
// ============================================================

#import "@preview/cetz:0.3.4": canvas, draw

= Methodology

== Introduction

This chapter documents the complete empirical pipeline implemented in this study and
explains the reasoning behind each design decision. Every choice — from the choice of event
window to the selection of PCA over UMAP for dimensionality reduction — is grounded in a
concrete constraint: the data are sparse, the signal is weak, and the sample is small by the
standards of modern machine learning. Understanding those constraints is the key to
evaluating the pipeline's outputs fairly.

The chapter is organised as follows. @sec-philosophy positions the study philosophically
and explains why a deductive quantitative approach is the correct fit for the research
questions. @sec-data documents the data sources, sample filters, and the event-study
label construction that translates raw stock returns into a binary prediction target.
@sec-features describes how each of the three information modalities — financial ratios,
textual embeddings, and graph topology — is constructed into model-ready features.
@sec-architecture presents the full late-fusion architecture with an annotated diagram and
explains the key design trade-offs. @sec-evaluation specifies the ablation ladder, the three
hypothesis tests, and the evaluation metrics. @sec-limitations acknowledges the study's
known constraints openly. @sec-ethics closes with reproducibility and ethical
considerations.

== Research Philosophy and Design Logic <sec-philosophy>

=== Philosophical Positioning

The study adopts a post-positivist research philosophy @creswell2014. Post-positivism sits
between the strict objectivism of classical positivism and the constructivist view that
reality is entirely observer-dependent. It holds that the phenomenon of interest — here,
post-merger synergy — exists as a real economic outcome, but that any measurement of it
will be imperfect. Stock price reactions to deal announcements are real signals of market
expectation, but they are contaminated by behavioural bias, information asymmetry, and the
structural noise of financial markets @akerlof1970 @roll1986 @martynova2008. The
appropriate response is probabilistic inference: the study asks whether the proposed
architecture improves discrimination between value-creating and value-destroying deals, not
whether it recovers a noise-free synergy quantity.

The epistemological stance is deductive. Each of the three hypotheses was derived from a
specific structural failure identified in the literature review: the absence of topological
information (H1), the conflation of semantically opposite textual signals (H2), and the
relationship between network position and outcome variance (H3). Chapter 3's role is to
translate those theoretical claims into a controlled empirical pipeline capable of subjecting
them to falsifiable statistical tests @mackinlay1997 @betton2008.

=== Quantitative Empirical Strategy

The study combines two methodological traditions that are typically kept separate:
event-study finance and machine learning. Event studies, formalised by @mackinlay1997,
provide the prediction target: a binary label derived from the abnormal stock return earned
by the acquirer around the deal announcement. Machine learning provides the feature-mapping
machinery that translates three heterogeneous information sources into a discriminative
prediction. Classical econometrics defines what to predict; machine learning determines how
to predict it.

This hybrid is necessary because neither tradition alone is sufficient. A pure event-study
regression on financial ratios inherits all of Stream II's structural ceiling; a pure ML model
without an economically grounded target variable risks fitting noise rather than signal. The
combination grounds the prediction objective in well-understood financial theory while
exploiting the representational flexibility of modern learned models.

== Data, Sampling, and Target Construction <sec-data>

=== Data Sources

Four data environments are integrated, each addressing a specific theoretical gap identified
in the literature review. @tbl-datasources summarises the mapping between data source and
theoretical motivation.

#figure(
  table(
    columns: (1.5fr, 1.5fr, 2fr),
    align: (left, left, left),
    stroke: (x, y) => if y == 0 { (bottom: 1pt + black, top: 1pt + black) } else { none },
    inset: 8pt,
    table.header(
      [*Data Source*], [*What It Provides*], [*Theoretical Motivation*],
    ),
    [LSEG Workspace / Refinitiv], [Deal universe, transaction variables, deal structure flags], [Core label construction; acquirer / target identity; deal-level features],
    [CRSP market return series], [Daily stock returns for acquirers and market benchmarks], [Event-study CAR computation following @mackinlay1997],
    [Bloomberg SPLC], [Inter-firm supply chain and competition relationships], [Graph Block C; topological features for H1 and H3],
    [SEC EDGAR (10-K filings)], [Annual MD&A and Risk Factors text for acquirer and target], [Textual Block B; semantic features for H2],
  ),
  caption: [Data sources and their theoretical motivation.],
) <tbl-datasources>

Each source exists because a specific omission in prior work was diagnosed in Chapter 2 and
then deliberately addressed. The study is therefore theory-indexed rather than
data-driven: the data collection pipeline was designed around the hypotheses, not the
other way around.

=== Sample Construction

The analytical sample covers completed M&A deals involving publicly listed U.S. acquirers
between 2010 and 2023. This window captures the post-financial-crisis regulatory
environment while providing sufficient temporal depth for chronologically ordered
train / validation / test splits. Three exclusion criteria are applied:

+ Deals without at least 120 trading days of pre-announcement return history are removed because the market model estimation window would be too short to produce stable beta estimates.
+ Deals in which the acquirer is the same entity as the target (internal restructurings) are excluded because the event-study framework requires two distinct market entities.
+ For modality-specific ablation experiments, deals lacking the required data source (EDGAR filings or SPLC graph data) are excluded only from the models that require that modality, preserving sample size for the financial-only baseline.

The resulting sample size is approximately 4,999 deals for the financial-only models,
shrinking to approximately 1,140 deal-pairs with complete textual coverage and
approximately 2,864 deals with matched graph centrality data.

=== Event Study Label Design <sec-label>

The binary prediction target is derived from the Cumulative Abnormal Return (CAR) earned
by the acquirer's stock over the eleven-day window surrounding the deal announcement. This
process involves three steps.

*Step 1 — Estimate Normal Returns.* For each acquirer $i$, a single-factor market model is
estimated over the pre-event window $[-250, -10]$ trading days relative to announcement:
$ R_(i,t) = hat(alpha)_i + hat(beta)_i R_(m,t) + epsilon_(i,t) $
where $R_(m,t)$ is the value-weighted CRSP market return. The estimated parameters
$hat(alpha)_i$ and $hat(beta)_i$ represent the stock's typical return pattern in the absence of
deal-specific news.

*Step 2 — Compute Abnormal Returns.* For each day $t$ in the event window, the abnormal
return is the gap between the stock's actual return and the return predicted by the market
model:
$ "AR"_(i,t) = R_(i,t) - (hat(alpha)_i + hat(beta)_i R_(m,t)) $

*Step 3 — Cumulate and Label.* The eleven daily abnormal returns are summed across the
$[-5, +5]$ window:
$ "CAR"_i = sum_(t = -5)^(+5) "AR"_(i,t) $
The binary label is then:
$ y_i = cases(1 "if" "CAR"_i > 0 quad "(value-creating)", 0 "otherwise" quad "(value-destroying)") $

Predicting sign rather than magnitude is a deliberate choice. The magnitude of a CAR is
shaped by dozens of simultaneous factors — competing bids, macro shocks, payment method
surprises — that are effectively random from the perspective of pre-announcement
observables. The sign of the market's reaction, by contrast, is more systematically linked
to the underlying deal quality: whether the acquirer has the financial capacity, the strategic
alignment, and the ecosystem positioning to generate value. This is precisely the
information the three modalities are designed to capture.

The event window of $[-5, +5]$ is chosen because M&A information routinely diffuses before
the formal announcement through analyst reports, rumours, and abnormal pre-bid trading.
Robustness checks using the narrower $[-1, +1]$ window are reported to confirm that the
directional findings are not an artefact of the wider horizon @betton2008.

== Modality-Specific Feature Construction <sec-features>

The three modalities map directly onto the three theoretical signal domains identified in
Chapter 2. @tbl-featureblocks summarises the construction logic for each block.

#figure(
  table(
    columns: (1fr, 1.5fr, 1.5fr, 1.5fr),
    align: (center, left, left, left),
    stroke: (x, y) => if y == 0 { (bottom: 1pt + black, top: 1pt + black) } else { none },
    inset: 8pt,
    table.header(
      [*Block*], [*Raw Input*], [*Construction Steps*], [*Final Representation*],
    ),
    [*A — Financial*], [LSEG ratios: leverage, liquidity, profitability, valuation, deal premium, payment method, relative size], [Median imputation → StandardScaler (fit on train only)], [56-dimensional dense vector per deal],
    [*B — Textual*], [EDGAR 10-K: MD&A section + Risk Factors section (acquirer and target separately)], [Section extraction → FinBERT tokenisation → CLS pooling → pairwise cosine distance → PCA compression (fit on train only)], [128-dimensional PCA vector per deal-pair (64 per section)],
    [*C — Graph*], [Bloomberg SPLC: supplier\_of, customer\_of, competitor\_of edges + acquires edges], [HeteroData construction → type-specific GraphSAGE (2 hops) → cross-type attention pooling → node embedding extraction for acquirer and target], [64-dimensional graph embedding per firm node],
  ),
  caption: [Feature construction summary across the three modality blocks.],
) <tbl-featureblocks>

=== Block A: Financial Features

The financial block encodes the acquirer's balance-sheet quality, the target's valuation
characteristics, the deal structure, and the market context at announcement. Variables
include leverage ratios, liquidity metrics, profitability margins, Tobin's Q, acquisition
premium, payment method (cash / stock / mixed), and relative deal size. All ratios are
sourced from the fiscal year-end preceding the announcement date, which prevents any
forward-looking leakage from post-announcement filings.

The financial block is not where this study claims novelty. Its role is to establish a serious
benchmark: if the graph and text modalities do not add discriminative power over a well-
constructed financial baseline, the multimodal thesis fails on its own terms. The block is
therefore designed to be comprehensive enough to constitute a genuine ceiling, not a
deliberately weak comparison.

=== Block B: Textual Features

Each 10-K filing is split into its constituent sections before any embedding is extracted.
This section split is central to H2: the Management Discussion & Analysis (MD&A) section,
where management describes strategy and growth plans, and the Risk Factors section, where
the company enumerates its specific regulatory and operational hazards, encode opposite
hypothesised signals. MD&A similarity between acquirer and target is predicted to correlate
positively with CAR (strategic alignment); Risk Factor similarity is predicted to correlate
negatively (shared vulnerability / risk concentration). These are *opposite signals* — one
pushes the prediction toward positive outcome, the other toward negative. A model that
conflates these sections by embedding the entire filing as one block mixes these opposite signals into a single feature, which Chapter 4 empirically demonstrates
destroys predictive value.

For each section and each firm, the text is tokenised and passed through frozen FinBERT
@araci2019 weights to produce a 768-dimensional CLS embedding. Pairwise cosine
similarity is then computed between the acquirer's and target's embeddings within each
section, yielding two scalar similarity scores per deal. These scalars are used directly in
the H2 OLS test. For the fusion model, the full embeddings are PCA-compressed to 64
dimensions per section — a deliberate regularisation choice. PCA is the only dimensionality
reduction technique that produces a deterministic, out-of-sample-transformable basis,
making it compatible with rigorous fold-level cross-validation; non-linear alternatives such
as UMAP produce sample-dependent transformations that cannot be applied to held-out folds
without information leakage.

=== Block C: Graph Features

The graph block represents the corporate ecosystem as a heterogeneous directed graph
$cal(G) = (cal(V), cal(E), cal(T)_v, cal(T)_e)$, where nodes are firms and edges carry one of four
semantic types: `supplier_of`, `customer_of`, `competitor_of`, and `acquires`. The
heterogeneous structure is essential: a `supplier_of` edge implies operational dependency and
risk propagation, while a `competitor_of` edge implies market concentration and pricing power
dynamics. Collapsing these into a single undirected edge type would average out semantically
distinct economic mechanisms.

GraphSAGE @hamilton2017 is chosen over transductive GCN variants because the M&A
problem has an inductive structure: new deals involve firms that may not have been present
in the training graph. GraphSAGE learns aggregation functions over sampled neighbourhoods
rather than memorising node-specific embeddings, making it applicable to firms it has not
seen during training. Type-specific aggregation functions are applied independently per edge
type across two message-passing hops before a cross-type attention layer pools the
relational signals into a single 64-dimensional node embedding per firm.

== Model Architecture and Late-Fusion Design <sec-architecture>

@fig-architecture presents the complete pipeline from raw data sources to binary
prediction output. The diagram is organised top-to-bottom: raw inputs enter from the top,
modality-specific encoders produce fixed-dimensional embedding vectors, and these are
concatenated and passed to a downstream XGBoost classifier that produces the final
probability estimate.

#figure(
  canvas({
    import draw: *

    // ── colour palette ──────────────────────────────────────────
    let fin_col   = rgb("#5591c7")   // blue   – financial
    let txt_col   = rgb("#fdab43")   // amber  – textual
    let grp_col   = rgb("#00c49f")   // teal   – graph
    let fuse_col  = rgb("#9b59b6")   // purple – fusion head
    let lbl_col   = rgb("#e74c3c")   // red    – output label
    let bg_col    = rgb("#f0f0f0")   // light grey – data source boxes

    // ── helper: rounded rectangle with label ─────────────────────
    let box(x, y, w, h, fill: white, stroke: black, label: "", sub: "", font-size: 9pt) = {
      rect((x - w/2, y - h/2), (x + w/2, y + h/2),
        fill: fill, stroke: stroke, radius: 0.15)
      if sub == "" {
        content((x, y), text(size: font-size, weight: "bold", label))
      } else {
        content((x, y + 0.15), text(size: font-size, weight: "bold", label))
        content((x, y - 0.2), text(size: 7.5pt, style: "italic", sub))
      }
    }

    // ── helper: arrow ────────────────────────────────────────────
    let arr(from, to, col: black) = {
      line(from, to, stroke: (paint: col, thickness: 1.2pt),
           mark: (end: (symbol: ">")))
    }

    // ─────────────────────────────────────────────────────────────
    // ROW 0: DATA SOURCE BOXES  (y = 8.5)
    // ─────────────────────────────────────────────────────────────
    box(-6, 8.5, 2.8, 0.7, fill: bg_col, label: "LSEG / Refinitiv",   sub: "Financial ratios & deal metadata")
    box( 0, 8.5, 2.8, 0.7, fill: bg_col, label: "SEC EDGAR 10-K",     sub: "MD&A + Risk Factors text")
    box( 6, 8.5, 2.8, 0.7, fill: bg_col, label: "Bloomberg SPLC",     sub: "Supply-chain & competition graph")

    // ─────────────────────────────────────────────────────────────
    // ROW 1: PRE-PROCESSING  (y = 7.0)
    // ─────────────────────────────────────────────────────────────
    box(-6, 7.0, 2.8, 0.65, fill: rgb("#dbe9f8"), stroke: fin_col,
        label: "Impute + Scale",
        sub: "Median imputation, StandardScaler")

    box( 0, 7.0, 2.8, 0.65, fill: rgb("#fff3e0"), stroke: txt_col,
        label: "Section Split + Tokenise",
        sub: "MD&A vs Risk Factors extraction")

    box( 6, 7.0, 2.8, 0.65, fill: rgb("#e0f5ee"), stroke: grp_col,
        label: "HeteroData Graph Build",
        sub: "supplier / customer / competitor edges")

    // arrows row 0 → row 1
    arr((-6, 8.15), (-6, 7.35), col: fin_col)
    arr((0,  8.15), (0,  7.35), col: txt_col)
    arr((6,  8.15), (6,  7.35), col: grp_col)

    // ─────────────────────────────────────────────────────────────
    // ROW 2: ENCODERS  (y = 5.5)
    // ─────────────────────────────────────────────────────────────
    box(-6, 5.5, 2.8, 0.8, fill: fin_col, stroke: none,
        label: "Block A — Financial",
        sub: "56-d dense feature vector",
        font-size: 9pt)

    // textual: two sub-paths
    box(-1.2, 5.5, 2.4, 0.75, fill: txt_col, stroke: none,
        label: "FinBERT (MD&A)",
        sub: "768-d CLS → PCA 64-d")

    box( 1.2, 5.5, 2.4, 0.75, fill: txt_col, stroke: none,
        label: "FinBERT (Risk Factors)",
        sub: "768-d CLS → PCA 64-d")

    box(6, 5.5, 2.8, 0.8, fill: grp_col, stroke: none,
        label: "Block C — HeteroGraphSAGE",
        sub: "2-hop agg → 64-d node emb",
        font-size: 8pt)

    // arrows row 1 → row 2
    arr((-6,  6.68), (-6, 5.92), col: fin_col)
    arr((0,   6.68), (-1.2, 5.92), col: txt_col)
    arr((0,   6.68), ( 1.2, 5.92), col: txt_col)
    arr((6,   6.68), (6,   5.92), col: grp_col)

    // bracket to join the two FinBERT boxes into Block B
    line((-2.4, 5.1), (2.4, 5.1), stroke: (paint: txt_col, thickness: 1pt, dash: "dashed"))
    content((0, 4.85), text(size: 8pt, fill: txt_col, weight: "bold", "Block B — Textual (128-d total)"))

    // ─────────────────────────────────────────────────────────────
    // ROW 3: ANTI-LEAKAGE NOTE  (y = 4.3)
    // ─────────────────────────────────────────────────────────────
    rect((-7.5, 4.0), (7.5, 4.55),
         fill: rgb("#fff9e6"), stroke: (paint: rgb("#f0a500"), thickness: 0.8pt, dash: "dashed"),
         radius: 0.1)
    content((0, 4.28), text(size: 8pt, style: "italic",
      "⚠  All scaling, imputation, and PCA bases are fit exclusively on training folds — never on held-out data"))

    // ─────────────────────────────────────────────────────────────
    // ROW 4: CONCATENATION  (y = 3.2)
    // ─────────────────────────────────────────────────────────────
    box(0, 3.2, 5.5, 0.7, fill: fuse_col, stroke: none,
        label: "Late Fusion: Concatenate  [ h_F ‖ h_T ‖ h_G ]",
        sub: "(56 + 128 + 64 = 248 dimensions)",
        font-size: 8.5pt)

    // convergence arrows from encoders → concat
    arr((-6, 5.1), (-2.6, 3.58), col: fin_col)
    arr((0, 4.7),  (0,   3.58), col: txt_col)
    arr((6, 5.1),  (2.6, 3.58), col: grp_col)

    // ─────────────────────────────────────────────────────────────
    // ROW 5: CLASSIFIER  (y = 2.1)
    // ─────────────────────────────────────────────────────────────
    box(0, 2.1, 5.5, 0.7, fill: fuse_col, stroke: none,
        label: "XGBoost Prediction Head",
        sub: "5-fold stratified CV  |  AUC-ROC primary metric")

    arr((0, 2.85), (0, 2.48))

    // ─────────────────────────────────────────────────────────────
    // ROW 6: OUTPUT  (y = 1.0)
    // ─────────────────────────────────────────────────────────────
    box( -2.0, 1.0, 3.0, 0.65, fill: lbl_col, stroke: none,
        label: "y = 1", sub: "Value-Creating (CAR > 0)")

    box(  2.0, 1.0, 3.0, 0.65, fill: rgb("#2ecc71"), stroke: none,
        label: "y = 0", sub: "Value-Destroying (CAR ≤ 0)")

    arr((0, 1.75), (-2.0, 1.35), col: lbl_col)
    arr((0, 1.75), ( 2.0, 1.35), col: rgb("#2ecc71"))

    // ─────────────────────────────────────────────────────────────
    // ROW 7: SHAP INTERPRETABILITY LAYER
    // ─────────────────────────────────────────────────────────────
    rect((-5.0, 0.1), (5.0, 0.65),
         fill: rgb("#f4f4f4"), stroke: (paint: rgb("#7f8c8d"), thickness: 0.8pt),
         radius: 0.1)
    content((0, 0.38), text(size: 8pt,
      "SHAP decomposition: per-modality attribution → economic credibility layer"))

    arr((0, 0.72), (0, 1.35), col: rgb("#7f8c8d"))

  }),
  caption: [Full multimodal architecture — from raw data sources to binary CAR prediction, with anti-leakage guarantee and SHAP interpretability layer.],
) <fig-architecture>

=== Why Late Fusion Was Chosen

The architecture diagram shows that each encoder is trained independently before the
outputs are concatenated. This decoupled approach — called *late fusion* — contrasts with
*early fusion* (concatenate raw features from all modalities before any encoding) and *joint
end-to-end training* (train all encoders simultaneously on the final CAR label).

Joint end-to-end training was considered and rejected for a concrete reason: the number of
complete multimodal observations (approximately 1,140–2,864 depending on modality) is
far below the sample sizes typically required for stable simultaneous fine-tuning of
transformer and GNN components @baltrusaitis2019. With a noisy binary label, a weak
class signal, and fewer than 5,000 observations, joint training would most likely produce
memorisation rather than generalisation. Late fusion isolates the representation learning
within each modality from the cross-modal inference task, allowing each encoder to produce
stable, reusable representations even when modality coverage is incomplete.

=== Inductive Transfer Learning Framing

FinBERT and GraphSAGE were not pre-trained to predict M&A synergy — they encode
general financial language structure and general relational structure respectively. The
architecture treats this as an inductive transfer problem: the upstream encoders produce
reusable semantic and topological priors; the downstream XGBoost layer performs the
task-specific mapping from those priors to CAR direction. The validity of this transfer is tested directly and confirmed through the ablation ladder in @sec-evaluation and SHAP attribution in Chapter 4.

=== Why SHAP Is Methodologically Central

SHAP (SHapley Additive exPlanations) @lundberg2017 provides a game-theoretic
decomposition of each feature's marginal contribution to individual predictions. In this
architecture, SHAP serves a specific evidentiary function: it tests whether the frozen
textual and graph embeddings contribute explanatory mass beyond the financial features.
If graph and text components never appear among the dominant SHAP contributors, the AUC
improvement over the financial baseline would lack economic credibility — it could reflect
a statistical artefact rather than genuine multimodal signal. Conversely, their consistent
appearance among the top contributors constitutes evidence that the transferred embeddings
encode synergy-relevant structure. SHAP is therefore not an optional interpretability
add-on; it is the mechanism by which the architecture's claims are substantiated.

== Evaluation Design and Hypothesis Testing <sec-evaluation>

=== The Ablation Ladder

Models are evaluated in a structured ablation sequence rather than as a single headline
comparison. @tbl-ablation-ladder maps each model configuration to its name, feature count,
and the specific research claim it tests.

#figure(
  table(
    columns: (1fr, 2fr, 1fr, 2.5fr),
    align: (center, left, center, left),
    stroke: (x, y) => if y == 0 { (bottom: 1pt + black, top: 1pt + black) } else { none },
    inset: 8pt,
    table.header(
      [*Model*], [*Description*], [*Features*], [*Purpose*],
    ),
    [M1], [Financial Only], [56], [Tabular ceiling — benchmark any successor must beat],
    [M2], [Financial + Text (unsplit)], [184], [Demonstrates cost of section conflation — the M2 Reversal test],
    [M3], [Financial + Text (section-split) + Graph], [248], [Full multimodal model — primary headline result],
    [M3e], [M3 + engineered interaction features], [261], [Sensitivity to hand-crafted features vs learned representations],
  ),
  caption: [Ablation ladder: model configurations and their role in the experimental design.],
) <tbl-ablation-ladder>

The ladder structure prevents the thesis from hiding behind a monolithic final model: every
AUC gain must be attributable to a specific modality, and the source of each gain must be
confirmed through both the ablation comparison and the SHAP decomposition.

=== Hypothesis Tests

@tbl-hypothesis-tests maps each hypothesis to its operationalisation, the statistical test
applied, and the evidence threshold for support.

#figure(
  table(
    columns: (1fr, 2.5fr, 1.5fr, 2fr),
    align: (left, left, left, left),
    stroke: (x, y) => if y == 0 { (bottom: 1pt + black, top: 1pt + black) } else { none },
    inset: 8pt,
    table.header(
      [*Hypothesis*], [*Claim*], [*Statistical Test*], [*Evidence Threshold*],
    ),
    [H1 — Topological Alpha], [GraphSAGE embeddings (Block C) produce a statistically significant AUC-ROC increase over the financial-only baseline, disproportionately in supply-chain-intensive sectors (SIC 20–49)], [Paired $t$-test on 5-fold CV AUC scores; sector-stratified AUC comparison], [$p < 0.05$; M3 AUC $>$ M1 AUC; SHAP graph features in top-20],
    [H2 — Semantic Divergence], [MD&A section similarity between acquirer and target positively predicts CAR ($beta_"MDA" > 0$); Risk Factor similarity negatively predicts CAR ($beta_"RF" < 0$)], [Bivariate OLS: $"CAR"_i = beta_0 + beta_1 dot "sim"_"MDA,i" + beta_2 dot "sim"_"RF,i" + epsilon_i$], [Sign asymmetry confirmed; $beta_"MDA" > 0$ and $beta_"RF" < 0$],
    [H3 — Topological Arbitrage], [Acquirers with high betweenness centrality exhibit compressed variance in $|"CAR"|$ relative to peripheral acquirers], [Levene's test for equality of variance across betweenness centrality quantile groups], [$p < 0.05$ on Levene's $F$; negative correlation between centrality and $|"CAR"|$],
  ),
  caption: [Hypothesis operationalisation: claims, statistical tests, and evidence thresholds.],
) <tbl-hypothesis-tests>

=== Evaluation Metrics

AUC-ROC is the primary metric because it is threshold-invariant and robust to class
imbalance — both relevant properties given that the CAR sign split is not guaranteed to be
50/50 across all sub-samples. Accuracy and F1 are reported as secondary diagnostics.
Continuous regression targets (predicting CAR magnitude rather than direction) are tested
as a robustness check; as demonstrated in Chapter 4, linear regression on CAR magnitude
produces negative $R^2$ across all model configurations, confirming that binary directional
classification is the appropriate problem framing @fama1991.

== Limitations <sec-limitations>

Every empirical study involves trade-offs between the ideal design and what the available
data and resources make feasible. @tbl-limitations maps each limitation to its source and
the mitigation applied.

#figure(
  table(
    columns: (2fr, 2fr, 2.5fr),
    align: (left, left, left),
    stroke: (x, y) => if y == 0 { (bottom: 1pt + black, top: 1pt + black) } else { none },
    inset: 8pt,
    table.header(
      [*Limitation*], [*Source*], [*Mitigation Applied*],
    ),
    [Event window $[-5,+5]$ may capture confounding non-deal news], [Wider windows include days unrelated to the announcement], [Robustness checks with $[-1,+1]$ confirm directional results are not window-specific],
    [Single-factor market model may under-adjust for style premia in smaller acquirers], [Omission of Fama-French SMB / HML factors], [Binary direction target reduces sensitivity to magnitude errors; simplification acknowledged rather than concealed],
    [PCA discards non-linear geometry in FinBERT embeddings], [Linear projection applied to non-linear manifold], [PCA chosen for deterministic out-of-sample transformability; UMAP / t-SNE incompatible with fold-level CV],
    [Frozen FinBERT / GraphSAGE embeddings not supervised on CAR], [Transfer misalignment between pre-training objective and prediction task], [Ablation ladder and SHAP attribution test whether transfer succeeds empirically; no assumption made],
    [Sample size ($n approx 1,140$ to $4,999$) is small by deep learning standards], [Data availability constraints on complete multimodal coverage], [Late fusion isolates each encoder from the noisy label; XGBoost chosen for its documented advantage at M&A-scale tabular inputs],
  ),
  caption: [Known limitations, their source, and the mitigation strategy applied.],
) <tbl-limitations>

== Ethics, Reproducibility, and Implementation Finality <sec-ethics>

All data used in this study are obtained through institutional and public channels. No
personal data, no individual-level human subject data, and no sensitive commercial
information are collected or processed beyond what is publicly disclosed in regulatory filings
and commercial financial databases. The principal ethical obligations are licensing
compliance, accurate reporting of model performance, and honest acknowledgement of
limitations — all of which are addressed throughout this chapter.

Reproducibility is ensured by the frozen implementation. The GitHub repository contains
the final codebase used to generate all results reported in Chapter 4: the preprocessing
pipeline, the event-study label constructor, the FinBERT embedding extractor, the
HeteroGraphSAGE encoder, the late-fusion XGBoost classifier, and the SHAP attribution
scripts. This methodology chapter describes that exact pipeline — not a simplified or
idealised version. Every design choice documented here corresponds to a concrete line of
code in the repository, and every reported metric corresponds to a stored cross-validation
result file.

#bibliography("works-methodology.bib", style: "ieee")
