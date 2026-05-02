// ============================================================
//  03-methodology.typ
//  Verified against: src/features/text.py, scripts/graphs/train_hetero_graph.py,
//  scripts/data/pull_car_data.py, scripts/data/compute_car.py,
//  src/models/fusion.py, scripts/training/training_utils.py
// ============================================================

#let tbl-caption(body) = text(style: "italic", size: 9pt, body)
#let code-inline(body) = raw(body, lang: none)

= Methodology <ch-methodology>
#label("ch-litreview")


== Introduction

This chapter details the research design, architectural implementation, and
evaluation protocols employed to construct the proposed tri-modal fusion model
for M&A synergy prediction.  The methodology is structured to operationalise
the theoretical findings from @ch-litreview, translating the need for
"multimodal fusion" into a rigorous engineering specification.

The chapter proceeds as follows. @sec-philosophy establishes the research
philosophy and epistemological stance. @sec-hypotheses formalises the three
hierarchical hypotheses.  @sec-data describes all data sources and collection
pipelines.  @sec-preprocessing covers cleaning, normalisation, and temporal
splitting.  @sec-features defines the three feature blocks (Financial,
Textual, Graph).  @sec-models specifies the baseline and fusion model
architectures.  @sec-car derives the CAR target variable.
@sec-htesting details the hypothesis-testing protocol, and
@sec-ethics addresses limitations and ethical considerations.

== Research Philosophy and Design <sec-philosophy>

This study adopts a *post-positivist* epistemological stance, treating M&A
synergy as a latent, probabilistic construct approximated through market
reactions and structured inter-firm relationships.  While acknowledging that
markets are not perfectly efficient, the research operates within the
semi-strong form of the Efficient Market Hypothesis @fama1991, wherein
publicly available information --- financial fundamentals, regulatory filings,
and network topology --- constitutes a viable predictor signal.  The
overarching research design is *quantitative and deductive*: three _a priori_
hypotheses (H1, H2, H3) are specified before analysis and tested through
controlled ablation experiments.

The study employs a *cross-sectional observational design*.  Because M&A deals
are historical and non-repeatable, no experimental manipulation is possible;
causal inference is instead approximated through systematic covariate control,
ablation modelling, and statistical hypothesis testing, following standard
practice in empirical corporate finance @mackinlay1997.

An *Experimental Prototyping SDLC* governs the engineering programme.
Non-deterministic model outputs and stochastic training dynamics demand
reproducible seeding, versioned artefacts, and isolated ablation configurations
rather than a traditional waterfall build process.  Each experimental variant
is fully specified in a YAML configuration (e.g., #code-inline("full_fusion.yaml"))
that pins hyperparameters, random seeds, feature subsets, and evaluation splits.

== Research Hypotheses <sec-hypotheses>

Three hierarchical hypotheses structure the empirical programme.  Together they
form a logical escalation: H1 tests whether the graph stream adds signal above
financials; H2 tests whether the text stream adds signal; H3 tests whether all
three streams fused together outperform every individual modality.

#figure(
  table(
    columns: (2.5cm, 3cm, 9cm),
    align: (center, left, left),
    inset: 7pt,
    stroke: 0.5pt,
    table.header(
      [*ID*], [*Name*], [*Formal Statement*],
    ),
    [H1],
    [Topological Alpha],
    [Supply-chain network centrality metrics derived from Bloomberg SPLC carry
     statistically significant predictive signal for acquirer CAR, incremental
     to financial fundamentals alone ($Delta"MAE" > 0$ on held-out test set,
     $p < 0.05$ by Diebold--Mariano test).],

    [H2],
    [Semantic Divergence],
    [The cosine distance between acquirer and target FinBERT embeddings of their
     respective 10-K MD&A sections is a significant predictor of post-acquisition
     CAR ($p < 0.05$, Pearson/Spearman correlation; confirmed by ablation).],

    [H3],
    [Topological Arbitrage],
    [The full tri-modal fusion (Financial + Text + Graph) significantly
     outperforms all single-modality baselines on held-out CAR prediction as
     measured by MAE and $R^2$, with $p < 0.05$ and Cohen's $d > 0.2$.],
  ),
  caption: [Research Hypotheses],
) <tbl-hypotheses>

== Data Sources and Collection <sec-data>

=== M&A Deal Universe

The primary dataset is sourced from the *London Stock Exchange Group (LSEG)
Refinitiv* database, which provides deal-level financial attributes for
completed M&A transactions.  Five raw CSV exports are merged via
#code-inline("scripts/data/build_combined_dataset.py") into
#code-inline("data/interim/ma_combined.csv").

The deal universe is restricted to:

- Completed acquisitions of publicly listed US targets by publicly listed US
  acquirers.
- Transactions announced between 2000 and 2023.
- Deal values exceeding USD 50 million (sufficient market microstructure data
  for reliable CAR estimation).

These filters follow established practice @betton2008 and ensure a minimum of
120 trading days in the estimation window.

=== Equity Return Data

Daily equity returns for acquirer firms and the S&P 500 benchmark are retrieved
via *yfinance* (Bloomberg ticker conversion handled by
#code-inline("pull_car_data.py")).  Returns are aligned to deal announcement
dates and stored in a long-format time series
(#code-inline("timeseries_long.csv")) with a #code-inline("rel_day") field
denoting trading-day distance from announcement (Day 0 = first trading day on
or after the announcement date, forward-fill rule).  Failed ticker lookups are
retried with fuzzy-matching heuristics.

=== Textual Data (SEC EDGAR)

10-K annual filings for acquirer firms are retrieved from the *SEC EDGAR
full-text search API*, targeting the MD&A (Item 7,
#code-inline("item_7_mda.txt")) and Risk Factors (Item 1A,
#code-inline("item_1a_risk.txt")) sections for the fiscal year immediately
preceding each announcement.  Extraction is handled by
#code-inline("src/features/edgar.py"), with download provenance logged in
#code-inline("data/external/edgar/download_log.csv").

=== Supply-Chain Network Data

Inter-firm supply-chain relationships are sourced from *Bloomberg SPLC*
(Supply Chain Analysis), which maps disclosed customer--supplier relationships
for publicly listed firms.  The SPLC data is merged with the deal universe via
#code-inline("scripts/data/merge_splc_data.py"), matching on Bloomberg ticker
symbols.  This forms the edge set for the heterogeneous graph constructed in
@sec-block-c.

== Data Preprocessing <sec-preprocessing>

=== Cleaning and Quality Control

Raw LSEG exports undergo systematic cleaning in
#code-inline("scripts/data/data_cleaning.py"): date parsing and
standardisation; deduplication of records sharing the same
acquirer--target--announcement-date triplet; removal of records with missing
acquirer ticker or announcement date; and currency normalisation to USD using
period-end exchange rates.

=== Feature Engineering and Normalisation

Financial features comprise 56 ratio-level variables spanning acquirer and
target leverage, liquidity, profitability, and deal structure characteristics.
The preprocessing pipeline applies:

+ *Winsorisation* at the 1st and 99th percentile to bound outlier influence.
+ *Z-score standardisation* (zero mean, unit variance) computed on
  training-set statistics _only_, then applied to validation and test sets.
+ *Stratified temporal splitting* (70 / 15 / 15) by announcement year to
  prevent temporal leakage.

=== Temporal Splitting and Event-Window Embargo <sec-embargo>

Deals are sorted chronologically and partitioned into training (2000--2016),
validation (2017--2019), and test (2020--2023) sets based on announcement year.
This strict temporal ordering ensures the model never trains on information
post-dating the validation or test periods.

#figure(
  box(
    width: 100%,
    stroke: 0.5pt,
    inset: 12pt,
    radius: 4pt,
    [
      #set align(center)
      #text(weight: "bold")[Temporal Dataset Partition]

      #v(6pt)

      #table(
        columns: (3.5cm, 3cm, 3cm, 4cm),
        align: center,
        inset: 6pt,
        stroke: 0.4pt,
        table.header(
          [*Partition*], [*Years*], [*Split*], [*Role*],
        ),
        [Training],   [2000--2016], [70%], [Model fitting],
        [Validation], [2017--2019], [15%], [Hyperparameter tuning + early stopping],
        [Test],       [2020--2023], [15%], [Final held-out evaluation],
      )

      #v(8pt)

      #text(size: 9pt)[
        *Embargo:* An 11-trading-day gap is enforced at each temporal boundary.
        Any deal announced within 11 days of a split boundary is excluded from
        both adjacent partitions, preventing CAR event-window overlap
        (cf. López de Prado, 2018).
      ]
    ]
  ),
  caption: [Temporal splits and embargo design],
) <fig-temporal-split>

Concretely, because the event window spans $[-5, +5]$ trading days, two deals
whose announcement dates differ by fewer than 11 trading days share overlapping
market-return sequences in their CAR calculations.  If one such deal falls in
the training set and the other in validation, the model can implicitly learn
return correlations that exist only because of calendar proximity --- the
*Overlapping Outcomes* problem formalised by López de Prado.
The 11-day embargo eliminates this cross-contamination by construction.

=== Missing Data Strategy

Features with $> 40%$ missing values are excluded.  For remaining missing
values, median imputation is applied to continuous features and mode imputation
to categorical indicators.  All imputation statistics are fitted on the
training set only.

== Feature Extraction <sec-features>

The three feature blocks are summarised in @tbl-featureblocks.

#figure(
  table(
    columns: (1.5cm, 3cm, 4.5cm, 5.5cm),
    align: (center, left, left, left),
    inset: 7pt,
    stroke: 0.5pt,
    table.header(
      [*Block*], [*Modality*], [*Source*], [*Construction (Fusion Pipeline)*],
    ),
    [A],
    [Financial],
    [LSEG Refinitiv],
    [56-column ratio matrix → Winsorise → z-score → #code-inline("ProjectionHead")
     ($RR^56 arrow.r RR^64$, linear + ReLU)],

    [B],
    [Textual\ (FinBERT)],
    [SEC EDGAR 10-K\ (Item 7 + Item 1A)],
    [FinBERT tokenisation (512-token chunks, stride = 256) →
     #code-inline("[CLS]") from penultimate layer → mean-pool across chunks →
     $RR^768$ per section →
     PCA compression (fit on train only): $RR^768 arrow.r RR^64$ per section →
     concatenate MD&A + RF vectors → $RR^128$ total.
     #linebreak()
     #text(style: "italic", size: 8.5pt)[Note: pairwise cosine similarity between
     acquirer and target section embeddings is computed separately as a scalar
     predictor for the H2 OLS test; it is not an input to the fusion model.]],

    [C],
    [Graph\ (HeteroGraphSAGE)],
    [Bloomberg SPLC],
    [HeteroConv (2-layer SAGEConv, separate per edge type) on
     #code-inline("(company, supplies, company)") and
     #code-inline("(company, buys_from, company)") edges →
     64-dim node embedding per firm.
     #linebreak()
     #text(style: "italic", size: 8.5pt)[Deals without SPLC coverage receive a
     zero vector; the graph stream is masked via
     #code-inline("has_graph = False") in the fusion model.]],
  ),
  caption: [Feature block definitions and fusion-pipeline construction steps.],
) <tbl-featureblocks>

=== Block A --- Financial Features <sec-block-a>

The financial feature vector $bold(h)_F in RR^(d_F)$ is constructed directly
from the standardised preprocessing output ($d_F = 56$).  For baseline models
(Ridge Regression, ElasticNet, XGBoost), $bold(h)_F$ is used directly.  For
the MLP and fusion models, it passes through a
#code-inline("ProjectionHead") --- a linear layer followed by ReLU --- that
maps $bold(h)_F$ to a lower-dimensional embedding
$hat(bold(h))_F in RR^64$ before concatenation.

=== Block B --- Textual Features (FinBERT) <sec-block-b>

Each acquirer firm's MD&A (Item 7) and Risk Factors (Item 1A) text is processed
through *FinBERT* (#code-inline("ProsusAI/finbert")) @araci2019, a
BERT-base architecture fine-tuned on financial communications corpora.  The
exact pipeline, as implemented in #code-inline("src/features/text.py"), is:

+ *Chunking.* The raw section text is tokenised (no truncation) and split into
  overlapping 512-token windows with a stride of 256 tokens, reserving
  positions 0 and 511 for the #code-inline("[CLS]") and #code-inline("[SEP]")
  tokens respectively.
+ *Extraction.* For each chunk, the #code-inline("[CLS]") token representation
  is taken from the *penultimate transformer layer*
  (#code-inline("hidden_states[-2]")), yielding a 768-dimensional vector.
+ *Pooling.* Chunk-level vectors are *mean-pooled* across all chunks to produce
  a single $bold(h)_T in RR^768$ per section.
+ *PCA compression.* Each section's embedding matrix is independently
  PCA-compressed:

$ bold(h)_T^("section") in RR^768 space arrow.r^("PCA, fit on train only") space bold(p)^("section") in RR^64 $

  Separate PCA models are fitted for MD&A and Risk Factors, serialised to
  #code-inline("data/processed/pca_models.pkl") for reproducible
  inference.  This compression reduces the $1536$-dimensional raw concatenation
  to $128$ dimensions while retaining maximum explained variance.

+ *Concatenation.* The two 64-dimensional section vectors are concatenated into
  the final textual embedding:

$ bold(h)_T = [bold(p)^"MDA" parallel bold(p)^"RF"] in RR^128 $

FinBERT's $approx 110$M parameters are *frozen* throughout all downstream
training to prevent overfitting given the limited M&A sample size; only the
downstream projection heads are trained.

==== Cosine Similarity (H2 Test Only)

For the H2 semantic-divergence hypothesis, a pairwise cosine similarity score
is computed *separately* between the acquirer and target's section embeddings,
_after_ PCA compression:

$ "SemanticDiv"_i = 1 - (bold(p)_"acq"^"MDA" dot bold(p)_"tgt"^"MDA") /
  (||bold(p)_"acq"^"MDA"|| dot ||bold(p)_"tgt"^"MDA"||) $

This scalar divergence score is used *exclusively* as the independent variable
in the H2 OLS regression.  It is *not* an input to the fusion model.  The
distinction is critical: cosine distance is a _relationship-level_ scalar
characterising strategic fit between two firms, while the fusion model requires
_firm-level_ vectors to learn independent acquirer representations.

=== Block C --- Graph Features (HeteroGraphSAGE) <sec-block-c>

The inter-firm supply-chain network is constructed as a heterogeneous graph
$cal(G) = (cal(V), cal(E), cal(T)_v, cal(T)_e)$ from Bloomberg SPLC data,
using PyTorch Geometric's #code-inline("HeteroData") object
(#code-inline("scripts/graphs/build_hetero_graph.py")).

==== Graph Structure

- *Node type.* A single node type #code-inline("company") represents each
  publicly listed firm present in the SPLC dataset.  Node features are
  initialised with degree centrality, betweenness centrality, and the
  standardised financial feature vector.

- *Edge types.* Two directed relationship types are encoded:

  #table(
    columns: (4cm, 4cm, 6cm),
    align: (center, center, left),
    inset: 6pt,
    stroke: 0.4pt,
    table.header(
      [*Edge Type*], [*Direction*], [*Semantics*],
    ),
    [#code-inline("supplies")],
    [supplier $arrow$ customer],
    [Firm A discloses Firm B as a customer in SPLC; directional dependency.],
    [#code-inline("buys_from")],
    [customer $arrow$ supplier],
    [Inverse of #code-inline("supplies"); encodes upstream procurement risk.],
  )

  All edges represent *currently active, disclosed supply-chain relationships*
  sourced from SPLC.  No M&A-derived edges (e.g., historical acquisition links)
  are included in the adjacency matrix.  This design choice eliminates any
  possibility of structural target leakage: the model has no graph path
  connecting an acquirer to its deal target, because no such edge class
  exists.

  #block(
    fill: luma(240),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
    [
      *Leakage Note.* The absence of acquisition edges is intentional and
      architecturally guarantees zero structural target leakage.  Supply-chain
      relationships exist independently of M&A outcomes and persist whether or
      not a deal completes.  The model cannot "see" the deal being predicted
      through the graph.

      Four edge types were considered in the project design phase
      (#code-inline("supplier_of"), #code-inline("customer_of"),
      #code-inline("competitor_of"), #code-inline("acquires")).  The
      implemented scope uses two SPLC-sourced types.  Competitor and historical
      acquisition edges are a natural extension discussed in @sec-ethics.
    ]
  )

==== HeteroGraphSAGE Model

A 2-layer *Heterogeneous GraphSAGE* model is trained via self-supervised link
prediction on the supply-chain graph, as implemented in
#code-inline("scripts/graphs/train_hetero_graph.py"):

#figure(
  box(
    width: 100%,
    stroke: 0.5pt,
    inset: 10pt,
    radius: 4pt,
    [
      #set align(center)
      #text(weight: "bold")[HeteroGraphSAGE Architecture]
      #v(4pt)
      #table(
        columns: (3cm, 5cm, 6cm),
        align: (center, center, left),
        inset: 6pt,
        stroke: 0.4pt,
        table.header(
          [*Layer*], [*Operation*], [*Detail*],
        ),
        [Conv 1],
        [#code-inline("HeteroConv")],
        [Separate #code-inline("SAGEConv(in → 128)") per edge type; mean aggregation across types],
        [Activation],
        [ReLU + Dropout],
        [$p = 0.3$; applied per-type after Layer 1],
        [Conv 2],
        [#code-inline("HeteroConv")],
        [Separate #code-inline("SAGEConv(128 → 64)") per edge type; mean aggregation],
        [Output],
        [Node embedding],
        [$bold(h)_G in RR^64$ per company node],
      )
      #v(6pt)
      #text(size: 8.5pt)[
        Training: self-supervised link prediction (binary cross-entropy),
        negative sampling per edge type, Adam lr = 0.01, 200 epochs.
        Final embeddings extracted via #code-inline("model.encode()") with full edge set.
      ]
    ]
  ),
  caption: [HeteroGraphSAGE architecture and training configuration.],
) <fig-hgnn-arch>

The key architectural innovation over a homogeneous GraphSAGE is that
#code-inline("supplies") and #code-inline("buys_from") edges learn *independent
SAGEConv weight matrices*, enabling the model to distinguish upstream procurement
signals from downstream customer dependency signals during message passing.
Node embeddings are extracted and stored in
#code-inline("data/interim/hetero_graph_embeddings.csv")
(64-dimensional, one row per company ticker), then merged into the training
dataset via deal--ticker matching.

== Model Architecture <sec-models>

=== Baseline Models

Four baselines are trained on Block A features only:

#figure(
  table(
    columns: (3cm, 3cm, 8.5cm),
    align: (center, center, left),
    inset: 7pt,
    stroke: 0.5pt,
    table.header(
      [*Model*], [*Variant ID*], [*Purpose*],
    ),
    [Naïve Mean],        [M0], [Lower bound; predicts training-set mean CAR for all deals.],
    [Ridge Regression],  [M1], [Linear baseline; controlled regularisation.],
    [ElasticNet],        [M2], [Feature selection; identifies redundant financial ratios.],
    [XGBoost],           [M3], [Non-linear financial-only ceiling; tests H1/H2/H3 incremental gain.],
    [Financial + Text],  [M4], [Ablation: removes graph stream; tests H2 in isolation.],
    [Financial + Graph], [M5], [Ablation: removes text stream; tests H1 in isolation.],
    [Full Fusion (F+T+G)],[M6], [Primary model; tests H3.],
  ),
  caption: [Model variants for ablation experiments.],
) <tbl-models>

=== Fusion Model Architecture <sec-fusion>

The primary model is the *late-fusion tri-modal architecture* implemented in
#code-inline("src/models/fusion.py").  Each active stream passes through its
own #code-inline("ProjectionHead") (linear + ReLU), and the resulting embeddings
are concatenated:

$ bold(z)_i = [bold(h)_F parallel bold(h)_T parallel bold(h)_G] in
  RR^(d_F' + d_T' + d_G') $

where $d_F' = 64$, $d_T' = 64$, $d_G' = 32$ by default.  The concatenated
vector $bold(z)_i$ is then passed through a two-layer MLP prediction head with
ReLU activation and dropout ($p = 0.3$) to produce the scalar CAR
prediction $hat(y)_i$.

#figure(
  box(
    width: 100%,
    stroke: 0.5pt,
    inset: 12pt,
    radius: 4pt,
    [
      #set align(center)
      #text(weight: "bold")[Tri-Modal Fusion Architecture]
      #v(8pt)

      // Stream labels
      #grid(
        columns: (1fr, 1fr, 1fr),
        gutter: 8pt,
        box(stroke: 0.5pt, inset: 8pt, radius: 3pt, width: 100%)[
          #align(center)[*Stream A*\ Financial\ $bold(h)_F in RR^56$]
          #v(3pt)
          #align(center)[↓ ProjectionHead]
          #align(center)[$hat(bold(h))_F in RR^64$]
        ],
        box(stroke: 0.5pt, inset: 8pt, radius: 3pt, width: 100%)[
          #align(center)[*Stream B*\ Textual\ $bold(h)_T in RR^128$]
          #v(3pt)
          #align(center)[↓ ProjectionHead]
          #align(center)[$hat(bold(h))_T in RR^64$]
        ],
        box(stroke: 0.5pt, inset: 8pt, radius: 3pt, width: 100%)[
          #align(center)[*Stream C*\ Graph\ $bold(h)_G in RR^64$]
          #v(3pt)
          #align(center)[↓ ProjectionHead]
          #align(center)[$hat(bold(h))_G in RR^32$]
        ],
      )

      #v(6pt)
      #align(center)[↓ Concatenate → $bold(z)_i in RR^160$]
      #v(4pt)
      #align(center)[↓ MLP Head (ReLU, Dropout $p=0.3$, 2 layers)]
      #v(4pt)
      #align(center)[↓ Scalar output: $hat(y)_i$ (predicted CAR)]

      #v(8pt)
      #text(size: 8.5pt)[
        Streams with missing data (#code-inline("has_graph=False")) contribute
        a zero vector.  The modular design enables controlled ablation by
        disabling any stream subset.
      ]
    ]
  ),
  caption: [Tri-modal late-fusion architecture (src/models/fusion.py).],
) <fig-fusion-arch>

=== Training Configuration

All PyTorch models are trained with:

- *Optimiser:* AdamW with cosine annealing learning-rate schedule with warm
  restarts.
- *Loss:* Mean Squared Error (MSE) as the primary objective.  Huber loss
  sensitivity analysis is included in evaluation.
- *Early stopping:* Validation MAE with patience of 15 epochs.
- *Batch size:* 64.
- *Reproducibility:* Fixed random seed via #code-inline("set_seed()") in
  #code-inline("src/training/trainer.py").
- *Device:* CUDA / Apple MPS / CPU auto-selected via
  #code-inline("src/config.py").

== Target Variable: Cumulative Abnormal Return <sec-car>

The target variable $y_i$ for each deal is the *Cumulative Abnormal Return
(CAR)* over the symmetric event window $[-5, +5]$ trading days relative to
announcement date, computed via the standard market model @brown1985
@mackinlay1997.  This section describes the full two-stage pipeline:
Stage 1 derives *actual* CAR values from raw market data using OLS; Stage 2
trains the fusion model to *predict* CAR from pre-announcement features and
evaluates predicted CAR against actual CAR on held-out deals.

#figure(
  box(
    width: 100%,
    stroke: 0.5pt,
    inset: 12pt,
    radius: 4pt,
    [
      #set align(center)
      #text(weight: "bold", size: 10pt)[Two-Stage CAR Pipeline]
      #v(10pt)
      #grid(
        columns: (1fr, 0.08fr, 1fr),
        gutter: 0pt,
        // Stage 1 box
        box(
          stroke: 0.6pt,
          inset: 10pt,
          radius: 4pt,
          width: 100%,
          fill: luma(248),
          [
            #align(center)[#text(weight: "bold")[Stage 1 — OLS Event Study]\ #text(size: 8pt)[(#raw("compute_car.py"))]]
            #v(6pt)
            #set text(size: 8.5pt)
            #set align(left)
            1. Download acquirer + S&P 500 daily prices (yfinance)\
            2. Compute log returns: $R_t = ln(P_t\/P_(t-1))$\
            3. Estimation window $[-200, -20]$: fit OLS\
            $quad R_(i t) = hat(alpha)_i + hat(beta)_i R_(m t)$\
            4. Event window $[-5, +5]$: compute AR\
            $quad A R_(i t) = R_(i t) - (hat(alpha)_i + hat(beta)_i R_(m t))$\
            5. Sum residuals:\
            $quad "CAR"_i = sum_(t=-5)^(+5) A R_(i t)$\
            6. Merge into #raw("final_car_dataset.csv")
          ]
        ),
        // Arrow column
        align(center + horizon)[
          #text(size: 20pt)[→]
          #v(4pt)
          #text(size: 7pt)[#raw("car_m5_p5")]
        ],
        // Stage 2 box
        box(
          stroke: 0.6pt,
          inset: 10pt,
          radius: 4pt,
          width: 100%,
          fill: luma(248),
          [
            #align(center)[#text(weight: "bold")[Stage 2 — Supervised Prediction]\ #text(size: 8pt)[(#raw("training_utils.py"))]]
            #v(6pt)
            #set text(size: 8.5pt)
            #set align(left)
            Input features (pre-announcement only):\
            $quad bold(z)_i = [bold(h)_F parallel bold(h)_T parallel bold(h)_G]$\
            \
            Fusion model predicts:\
            $quad hat(y)_i = f_theta (bold(z)_i)$\
            \
            Evaluated against actual CAR:\
            $quad y_i = "CAR"_i = $ #raw("car_m5_p5") column\
            \
            Loss (training):\
            $quad cal(L) = 1/N sum_i (y_i - hat(y)_i)^2$\
            \
            Held-out metrics:\
            $quad$ MAE, RMSE, $R^2$, Huber loss
          ]
        ),
      )
      #v(8pt)
      #text(size: 8pt, style: "italic")[
        Stage 1 outputs are fixed market-derived labels independent of any model.
        Stage 2 uses pre-announcement features only — no post-deal information enters $bold(z)_i$.
      ]
    ]
  ),
  caption: [The two-stage CAR pipeline: Stage 1 computes actual CAR via OLS event study; Stage 2 trains and evaluates the fusion model against those labels.],
) <fig-car-pipeline>

=== Stage 1: OLS Market Model <sec-ols>

The market model is estimated over the *estimation window* $[-200, -20]$
trading days (180-day window, minimum 120 valid observations) using OLS
as implemented in #code-inline("scripts/data/compute_car.py") via
#code-inline("scipy.stats.linregress"):

$ R_(i t) = alpha_i + beta_i R_(m t) + epsilon_(i t) $

where:
- $R_(i t) = ln(P_(i t) \/ P_(i,t-1))$ is the acquirer log return on trading day $t$,
- $R_(m t)$ is the S&P 500 (SPX) log return on the same day,
- $hat(alpha)_i$ is the estimated intercept (abnormal return in absence of market movement),
- $hat(beta)_i$ is the estimated systematic risk loading (market beta).

The OLS estimators are:

$ hat(beta)_i = ("Cov"(R_i, R_m)) / ("Var"(R_m)), quad hat(alpha)_i = bar(R)_i - hat(beta)_i bar(R)_m $

A gap window $[-19, -6]$ between estimation and event windows is excluded from
both calculations, preventing estimation-period price dynamics from contaminating
the event-window benchmark.

#figure(
  box(
    width: 100%,
    stroke: 0.5pt,
    inset: 10pt,
    radius: 4pt,
    [
      #set align(center)
      #text(weight: "bold")[Event Study Timeline (Trading Days)]
      #v(8pt)
      #table(
        columns: (3.5cm, 2.8cm, 3.2cm, 1.6cm, 3.2cm),
        align: center,
        inset: 6pt,
        stroke: 0.4pt,
        table.header(
          [*Estimation Window*], [*Gap (excl.)*], [*Day 0*], [*⟵*], [*Event Window*],
        ),
        [$[-200, -20]$\ OLS: fit $hat(alpha)_i, hat(beta)_i$],
        [$[-19, -6]$\ excluded],
        [*Announcement*\ (Day 0)],
        [],
        [$[-5, +5]$\ sum $A R_(i t)$ → $"CAR"_i$],
      )
    ]
  ),
  caption: [Event study timeline. The gap prevents estimation-period contamination of the CAR window.],
) <fig-event-timeline>

=== Stage 1: Abnormal Returns and CAR <sec-ar-car>

With $hat(alpha)_i$ and $hat(beta)_i$ estimated on the estimation window,
*abnormal returns* in the event window $cal(T) = {-5, ..., +5}$ are the
residuals between actual and model-predicted returns:

$ A R_(i t) = R_(i t) - (hat(alpha)_i + hat(beta)_i R_(m t)) $

$A R_(i t)$ represents the return attributable to deal-specific information
(announcement effect) after stripping out normal market co-movement.
*CAR is the cumulative sum of these residuals* over the full event window:

$ "CAR"_i = sum_(t=-5)^(+5) A R_(i t) $

This produces a single scalar per deal stored as column
#code-inline("car_m5_p5") in #code-inline("data/processed/final_car_dataset.csv").
A positive CAR indicates the market rewarded the acquisition announcement;
a negative CAR indicates value destruction.

=== Stage 2: Model Prediction vs. Actual CAR <sec-prediction-vs-actual>

The column #code-inline("car_m5_p5") (set as #code-inline("TARGET_COL") in
#code-inline("scripts/training/training_utils.py")) is the sole prediction
target for all model variants.  The fusion model is trained to minimise:

$ cal(L)(theta) = 1/N sum_(i=1)^N (y_i - hat(y)_i)^2 + lambda ||theta||_2^2 $

where $y_i = "CAR"_i$ (the Stage 1 OLS-derived actual CAR) and
$hat(y)_i = f_theta(bold(z)_i)$ (the fusion model's predicted CAR from
pre-announcement features only).  All evaluation metrics are then computed by
comparing $hat(y)_i$ against $y_i$ on the held-out test set:

#figure(
  table(
    columns: (3.5cm, 6cm, 5cm),
    align: (center, left, left),
    inset: 7pt,
    stroke: 0.5pt,
    table.header(
      [*Metric*], [*Formula*], [*Interpretation*],
    ),
    [MAE],
    [$frac(1,N) sum_i |y_i - hat(y)_i|$],
    [Primary metric; interpretable in percentage-point CAR terms.],

    [RMSE],
    [$sqrt(frac(1,N) sum_i (y_i - hat(y)_i)^2)$],
    [Penalises large mispredictions more than MAE.],

    [$R^2$],
    [$1 - frac(sum_i (y_i - hat(y)_i)^2, sum_i (y_i - bar(y))^2)$],
    [Proportion of CAR variance explained by the model.],

    [Huber],
    [$sum_i cal(H)_delta (y_i - hat(y)_i)$],
    [Robust to outlier returns; sensitivity analysis only.],

    [Dir. Accuracy],
    [$frac(1,N) sum_i bb(1)[text("sign")(hat(y)_i) = text("sign")(y_i)]$],
    [Practical deal advisory metric: did the model predict the sign correctly?],
  ),
  caption: [Evaluation metrics computed by comparing model-predicted CAR ($hat(y)_i$) against OLS-derived actual CAR ($y_i$).],
) <tbl-metrics>

The critical design guarantee is that *no feature in $bold(z)_i$ is derived
from post-announcement data*.  Financial ratios $bold(h)_F$ use the most recent
pre-announcement fiscal year; FinBERT embeddings $bold(h)_T$ use the 10-K filed
before announcement; graph embeddings $bold(h)_G$ use SPLC relationships that
are structurally independent of deal outcomes.  The model therefore predicts
market reaction from purely pre-deal information --- the economically meaningful
and leakage-free formulation.

== Hypothesis Testing <sec-htesting>

Each hypothesis is tested through model ablation combined with statistical
significance testing:

- *H1 (Topological Alpha):* Compare M5 (#code-inline("financial_graph.yaml"))
  vs. M3 (#code-inline("financial_only.yaml")).  A paired Diebold--Mariano test
  on hold-out prediction errors assesses whether the graph stream yields a
  statistically significant MAE improvement.

- *H2 (Semantic Divergence):* A Pearson/Spearman correlation test between
  #code-inline("SemanticDiv_i") and $"CAR"_i$ is run first.  Then M4
  (#code-inline("financial_text.yaml")) vs. M3 (#code-inline("financial_only.yaml"))
  ablation is evaluated with the same DM test.

- *H3 (Topological Arbitrage):* M6 (#code-inline("full_fusion.yaml")) is compared
  against all single-modality baselines.  Effect size (Cohen's $d$ on hold-out
  error distributions) and $R^2$ improvement are reported alongside $p$-values.

All tests use a significance threshold of $alpha = 0.05$ with *Bonferroni
correction* applied across the three hypothesis tests to control the
family-wise error rate ($alpha_"corrected" = 0.0167$).

== Evaluation Metrics <sec-metrics>

Primary metrics are *Mean Absolute Error (MAE)*, *Root Mean Squared Error
(RMSE)*, and *$R^2$* (coefficient of determination), all computed on the
held-out test set.  MAE is the primary metric given its interpretability in
percentage-point CAR terms.  Secondary evaluation includes a *directional
accuracy* metric (proportion of deals where predicted CAR sign matches actual
sign), which has practical relevance for deal advisory applications.

== Ethical Considerations and Limitations <sec-ethics>

All data used is commercially licensed (LSEG, Bloomberg) and contains no
personally identifiable information.  The study does not involve human
participants.

Key methodological limitations include:

+ *SPLC Disclosure Bias.* The supply-chain network captures only disclosed
  relationships, potentially biasing graph features toward larger firms with
  more extensive reporting obligations.  Smaller firms may have sparser
  neighbourhoods that understate their true network centrality.

+ *Frozen FinBERT Embeddings.* Frozen weights may not fully capture M&A-specific
  language not present in FinBERT's training corpus.  Fine-tuning on a
  domain-specific financial corpus represents a natural extension.

+ *Market-Model Beta Stationarity.* The OLS market model assumes beta is
  stationary over the estimation window, which may be violated for firms
  undergoing strategic repositioning pre-deal.

+ *US-Listed Sample.* The sample is restricted to US-listed firms, limiting
  generalisability to cross-border or private-equity transactions.

+ *Edge Type Scope.* The implemented graph uses two SPLC-derived edge types
  (#code-inline("supplies"), #code-inline("buys_from")).  Historical acquisition
  edges and competitor-of edges were considered in the design but not implemented
  within the project's data budget.  Their inclusion, using survivorship-bias-corrected
  datasets, is a natural direction for future work.
