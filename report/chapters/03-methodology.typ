// ============================================================
//  03-methodology.typ
// Chapter 3: Methodology
//  M&A Synergy Prediction | Hard Joshi | UEL
// ============================================================

#let tbl-caption(body) = text(style: "italic", size: 9pt, body)
#let code-inline(body) = raw(body, lang: none)




= Methodology <ch-methodology>
#show figure: set block(spacing: 1.5em)


== Introduction

The core objective of this methodology is to operationalise the theoretical requirement for multimodal fusion into a rigorous engineering specification for M&A synergy prediction. By integrating financial, textual, and topological features into a single heterogeneous framework, the study moves from treating companies as isolated data points to modelling them as networked economic actors. This design ensures that all architectural choices—from strict temporal partitioning to section-specific textual reasoning—serve the primary goal of isolating a genuine predictive edge while preventing forward-looking bias.

== Research Philosophy and Design <sec-philosophy>

#figure(
  box(
    width: 100%,
    stroke: 0.5pt,
    inset: 15pt,
    radius: 4pt,
    [
      #set align(center)
      #text(weight: "bold")[Multimodal M&A Intelligence System Architecture]
      #v(10pt)
      #grid(
        columns: (1fr, 1fr, 1fr),
        gutter: 10pt,
        box(stroke: 0.5pt, inset: 8pt, radius: 2pt, fill: luma(245))[
          *Data Sources*\ #v(5pt)
          #text(size: 8pt)[
            Deals: Yahoo Finance\
            Text: SEC EDGAR\
            Graph: Bloomberg SPLC
          ]
        ],
        box(stroke: 0.5pt, inset: 8pt, radius: 2pt, fill: luma(245))[
          *Feature Extraction*\ #v(5pt)
          #text(size: 8pt)[
            Financial Ratios\
            FinBERT Embeddings\
            GraphSAGE Nodes
          ]
        ],
        box(stroke: 0.5pt, inset: 8pt, radius: 2pt, fill: luma(245))[
          *Fusion & Evaluation*\ #v(5pt)
          #text(size: 8pt)[
            Late Fusion (MLP)\
            Ablation Testing\
            SHAP Attribution
          ]
        ],
      )
      #v(10pt)
      #text(size: 9pt)[→ Data flow from ingestion through to predictive inference →]
    ]
  ),
  caption: [System architecture pipeline showing the integration of Yahoo Finance deal data, SEC textual data, and Bloomberg topological data.],
) <fig-system-architecture>

This study adopts a *post-positivist* epistemological stance, treating M&A
synergy as a latent, probabilistic construct approximated through market
reactions and structured inter-firm relationships.  While acknowledging that
markets are not perfectly efficient, the research operates within the
semi-strong form of the Efficient Market Hypothesis @fama1970 @fama1991, wherein
publicly available information - financial fundamentals, regulatory filings,
and network topology - constitutes a viable predictor signal @jensen1978.  The
overarching research design is *quantitative and deductive*: three _a priori_
hypotheses (H1, H2, H3) are specified before analysis and tested through
controlled ablation experiments @creswell2014.

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
form a logical escalation: H1 tests whether the graph stream improves
directional discrimination (classification) above financials; H2 tests whether
pairwise textual similarity carries directionally opposing signal depending on
filing section; H3 tests whether graph centrality compresses the variance of
announcement returns via Levene's test across centrality quantile groups.

Predicting both the sign and the magnitude of CAR serves two distinct
analytical purposes.  Predicting the *magnitude* (via regression) is
notoriously difficult due to competing bids, information asymmetry, and macro
shocks, but it is necessary to test structural variance compression (H3).
Predicting the *sign* (via classification) is highly tractable and
systematically links to practical deal advisory: whether the deal is a net
positive or a net negative (H1, H2).  The architecture therefore formulates CAR
as a *dual target*: a continuous variable for variance analysis, and a
thresholded binary label ($1$ if $"CAR" > 0$, $0$ otherwise) for directional
discrimination.

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
     statistically significant predictive signal for acquirer CAR direction,
     incremental to financial fundamentals alone (statistically significant
     AUC-ROC improvement on held-out test set, $p < 0.05$ by paired
     $t$-test across cross-validation folds).],

    [H2],
    [Semantic Divergence],
    [The cosine distance between acquirer and target FinBERT embeddings of their
     respective 10-K MD&A sections is a significant predictor of post-acquisition
     CAR ($p < 0.05$, Pearson/Spearman correlation; confirmed by ablation).],

    [H3],
    [Topological Arbitrage],
    [Acquirer nodes with high betweenness centrality in the heterogeneous supply-chain graph will exhibit statistically compressed variance in $|"CAR"|$ outcomes relative to peripheral nodes. Success criterion: Levene's test (@levene1960) for equality of variances across centrality quantile groups yields $p < 0.05$, confirming the Information Transparency Dampening mechanism.],
  ),
  caption: [Research Hypotheses],
) <tbl-hypotheses>

== Data Sources and Collection <sec-data>

=== M&A Deal Universe

The primary dataset is sourced from *Yahoo Finance*, which provides deal-level financial attributes for completed M&A transactions.
completed M&A transactions.  Five raw CSV exports are merged via
#code-inline("scripts/data/build_combined_dataset.py") into
#code-inline("data/interim/ma_combined.csv").

The deal universe is restricted to:

- Completed acquisitions of publicly listed US targets by publicly listed US acquirers.
- Transactions announced between 2000 and 2023.
- Deal values exceeding USD 50 million (sufficient market microstructure data for reliable CAR estimation).

These filters follow established practice @betton2008 and ensure a minimum of 120 trading days in the estimation window. The systematic sample construction process is detailed in @tbl-sample-funnel.

#block(
  fill: luma(250),
  inset: 10pt,
  radius: 4pt,
  [
    *Research Note: The "Messy" Reality of Data Collection.*
    Building this sample was the most labor-intensive part of the project. Merging the Yahoo Finance deals with the Bloomberg SPLC graph was often frustrating; ticker symbols change, firms delist, and "matching" code frequently dropped hundreds of deals because of minor CSV formatting issues or trailing whitespaces. What looks like a clean funnel in @tbl-sample-funnel was actually several weeks of manual cross-referencing and debugging ticker-mapping logic in my terminal.
  ]
)

#figure(
  table(
    columns: (1fr, 0.5fr),
    align: (left, right),
    inset: 8pt,
    stroke: 0.5pt,
    fill: (x, y) => if y == 0 { luma(240) },
    table.header([*Stage*], [*Remaining Observations*]),
    [Raw M&A deals (Yahoo Finance / LSEG Refinitiv)], [4,999],
    [After filters (Date, Domestic, Completed)], [3,750],
    [With available announcement-day return data (yfinance)], [3,420],
    [With full financial fundamental coverage (Block A)], [3,180],
    [With SEC 10-K filing availability (Block B)], [2,921],
    [With SPLC graph node coverage (Block C)], [2,864],
    [*Final Modelling Sample (Full Multimodal Coverage)*], [*2,864*],
  ),
  caption: [Sample construction funnel demonstrating data attrition across filtration and modality-matching stages.],
) <tbl-sample-funnel>

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

Raw Yahoo Finance exports undergo systematic cleaning in
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
+ *Chronological holdout split* (70 / 15 / 15) by announcement year to
  prevent temporal leakage.

=== Temporal Splitting and Event-Window Embargo <sec-embargo>

Deals are sorted chronologically and partitioned into training (2000--2016),
validation (2017--2019), and test (2020--2023) sets based on announcement year.
This strict temporal ordering ensures the model never trains on information
post-dating the validation or test periods.

#figure(
  image("../../docs/figures/fig_temporal_split.png", width: 95%),
  caption: [Temporal partition design and 11-trading-day event-window embargo. Coloured blocks denote training (2000--2016, 70%), validation (2017--2019, 15%), and test (2020--2023, 15%) sets. The embargo gap at each boundary excludes any deal whose ±5-day CAR event window overlaps the partition boundary, eliminating the Overlapping Outcomes leakage mechanism formalised by #cite(<lopezdeprado2018>, form: "prose").],
) <fig-temporal-split>

Concretely, because the event window spans $[-5, +5]$ trading days, two deals
whose announcement dates differ by fewer than 11 trading days share overlapping
market-return sequences in their CAR calculations.  If one such deal falls in
the training set and the other in validation, the model can implicitly learn
return correlations that exist only because of calendar proximity --- the
*Overlapping Outcomes* problem formalised by #cite(<lopezdeprado2018>, form: "prose"). The 11-day embargo eliminates this cross-contamination by construction. A chronological holdout is the only valid evaluation design for temporal financial event data; random fold assignment would introduce macro-regime and return autocorrelation leakage that would invalidate all classification results.

=== Missing Data Strategy

Features with $> 40%$ missing values are excluded.  For remaining missing
values, median imputation is applied to continuous features and mode imputation
to categorical indicators.  All imputation statistics are fitted on the
training set only.

== Feature Extraction <sec-features>

The three feature blocks are summarised in @tbl-featureblocks.

#pagebreak(weak: true)
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
    [Yahoo Finance],
    [56-column ratio matrix → Winsorise → z-score → #code-inline("ProjectionHead")
     ($RR^56 → RR^64$, linear + ReLU)],

    [B],
    [Textual\ (FinBERT)],
    [SEC EDGAR 10-K\ (Item 7 + Item 1A)],
    [FinBERT tokenisation (512-token chunks, stride = 256) →
     #code-inline("[CLS]") from penultimate layer → mean-pool across chunks →
     $RR^768$ per section →
     PCA compression (fit on train only): $RR^768 → RR^64$ per section →
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

#figure(
  image("../../docs/figures/fig2_nlp_pipeline.jpg", width: 90%),
  caption: [Section-aware FinBERT NLP pipeline extracting MD&A and Risk Factors semantic vectors.],
) <fig-nlp-pipeline>

Each acquirer firm's MD&A (Item 7) and Risk Factors (Item 1A) text is processed
through *FinBERT* (#code-inline("ProsusAI/finbert")) #cite(<araci2019>, form: "prose"), a
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
  PCA-compressed as defined in @eq-pca:

$ bold(h)_T^("section") in RR^768 space →^("PCA, fit on train only") space bold(p)^("section") in RR^64 $ <eq-pca>

  Separate PCA models are fitted for MD&A and Risk Factors, serialised to
  #code-inline("data/processed/pca_models.pkl") for reproducible
  inference.  This compression reduces the $1536$-dimensional raw concatenation
  to $128$ dimensions while retaining maximum explained variance.

+ *Concatenation.* The two 64-dimensional section vectors are concatenated into
  the final textual embedding (@eq-text-concat):

$ bold(h)_T = [bold(p)^"MDA" parallel bold(p)^"RF"] in RR^128 $ <eq-text-concat>

FinBERT's $approx 110$M parameters are *frozen* throughout all downstream
training to prevent overfitting given the limited M&A sample size; only the
downstream projection heads are trained.

==== Cosine Similarity (H2 Test Only)

For the H2 semantic-divergence hypothesis, a pairwise cosine similarity score
is computed *separately* between the acquirer and target's section embeddings,
_after_ PCA compression:

$"SemanticDiv"_i = 1 - (p_"acq"^"MDA" dot p_"tgt"^"MDA") / (||p_"acq"^"MDA"|| dot ||p_"tgt"^"MDA"||)$ <eq-semantic-div> @salton1983

This scalar divergence score is used *exclusively* as the independent variable
in the H2 OLS regression.  It is *not* an input to the fusion model.  The
distinction is critical: cosine distance is a _relationship-level_ scalar
characterising strategic fit between two firms, while the fusion model requires
_firm-level_ vectors to learn independent acquirer representations.

=== Block C --- Graph Features (HeteroGraphSAGE) <sec-block-c>

#figure(
  image("../../docs/figures/fig3_heterographsage.jpg", width: 90%),
  caption: [HeteroGraphSAGE embedding process for supply-chain topological representation.],
) <fig-heterographsage>

The inter-firm supply-chain network is constructed as a heterogeneous graph
$cal(G) = (cal(V), cal(E), cal(T)_v, cal(T)_e)$ from Bloomberg SPLC data,
using PyTorch Geometric's #code-inline("HeteroData") object
(#code-inline("scripts/graphs/build_hetero_graph.py")).

==== Graph Structure

#figure(
  box(
    width: 100%,
    stroke: 0.5pt,
    inset: 15pt,
    radius: 4pt,
    [
      #set align(center)
      #text(weight: "bold")[Supply-Chain Graph Schema (Bloomberg SPLC)]
      #v(10pt)
      #grid(
        columns: (1fr, 0.5fr, 1fr),
        align: (center + horizon),
        box(stroke: 0.5pt, inset: 10pt, radius: 50%, fill: luma(245))[*Supplier*\ Node ($v_j$)],
        [#v(-10pt) → #text(size: 8pt)[supplies] #v(10pt) ← #text(size: 8pt)[buys_from]],
        box(stroke: 0.5pt, inset: 10pt, radius: 50%, fill: luma(245))[*Customer*\ Node ($v_i$)],
      )
      #v(10pt)
      #text(size: 9pt, style: "italic")[Heterogeneous edges: (company) —[supplies]→ (company)]
      #v(5pt)
      #table(
        columns: (1fr, 1fr),
        stroke: none,
        [*Node Features*], [*Edge Properties*],
        [Financial Ratios (56)], [Relationship Type],
        [Betweenness Centrality], [Revenue Exposure %],
        [Degree Centrality], [Yearly Validated],
      )
    ]
  ),
  caption: [Supply-chain graph construction showing inter-firm dependencies and network topology schema.],
) <fig-supply-chain-graph>

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
    [Inverse of #code-inline("supplies"); encodes upstream procurement risk. All edges represent *currently active, disclosed supply-chain relationships* sourced from SPLC. While the graph explicitly excludes M&A-derived edges (such as historical acquisition links) to reduce direct target leakage, it is important to note that SPLC relationships are not fully point-in-time archived. For earlier deals in the sample, the model may utilise network relationships that were only formalised after the deal announcement, introducing a residual look-ahead risk that is treated as a methodological limitation.],
  )

  All edges represent *currently active, disclosed supply-chain relationships* sourced from SPLC. While the graph explicitly excludes M&A-derived edges (such as historical acquisition links) to reduce direct target leakage, it is important to note that SPLC relationships are not fully point-in-time archived. For earlier deals in the sample, the model may utilise network relationships that were only formalised after the deal announcement, introducing a residual look-ahead risk that is treated as a methodological limitation.

  #block(
    fill: luma(240),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
    [
      *Leakage Note.* The absence of acquisition edges is intentional and
      reduces the risk of direct structural target leakage. While supply-chain
      relationships are theoretically independent of M&A outcomes, the use of
      static SPLC snapshots introduces a residual look-ahead risk for earlier 
      sample years. The model's graph coverage should therefore be interpreted 
      as an estimation of potential network signal under these data constraints.

      Four edge types were considered in the project design phase
      (#code-inline("supplier_of"), #code-inline("customer_of"),
      #code-inline("competitor_of"), #code-inline("acquires")).  The
      implemented scope uses two SPLC-sourced types.  Competitor and historical
      acquisition edges are a natural extension discussed in @sec-ethics.
    ]
  )

==== HeteroGraphSAGE Model

A 2-layer *Heterogeneous GraphSAGE* model #cite(<hamilton2017>, form: "prose") is trained via self-supervised link
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
    [Financial Only],    [M1], [Non-linear baseline (XGBoost); tests H1/H2 incremental gain.],
    [Financial + Text],  [M2], [Ablation: tests H2 in isolation; verifies text aggregation effect.],
    [Full Fusion (F+T+G)],[M3], [Primary model; tests H1.],
    [Full Fusion + Aux], [M3e], [Extended model with auxiliary engineered features.#footnote[*M3e extends M3 by appending 13 scalar auxiliary features to the concatenated embedding vector $bold(z)_i$: two deal-level derived features (relative deal size; cross-sector indicator), six graph-theoretic centrality scalars for acquirer and target nodes (betweenness centrality, degree centrality, PageRank), three pairwise textual similarity scalars (MD&A cosine similarity, Risk Factor cosine similarity, cross-section divergence), one acquirer-to-target asset size ratio, and the payment method indicator. These scalars are computed prior to the ProjectionHead and appended directly to $bold(z)_i in RR^(261)$. M3e does not improve AUC beyond M3 (0.5585 vs 0.5655), confirming that scalar centrality and similarity features add classification noise rather than signal when graph neighbourhood embeddings are already present.*]],
  ),
  caption: [Model variants for ablation experiments.],
) <tbl-models>

=== Fusion Model Architecture <sec-fusion>

#figure(
  box(
    width: 100%,
    stroke: 0.5pt,
    inset: 15pt,
    radius: 4pt,
    [
      #set align(center)
      #text(weight: "bold")[Multimodal Late-Fusion Architecture]
      #v(10pt)
      #grid(
        columns: (1fr, 1fr, 1fr),
        gutter: 15pt,
        box(stroke: 0.5pt, inset: 8pt, radius: 2pt, fill: luma(250))[
          *Block A (Fin)*\ $RR^56$\ #v(4pt) ↓\ #text(size: 9pt)[ProjHead]\ #v(4pt) ↓\ $RR^64$
        ],
        box(stroke: 0.5pt, inset: 8pt, radius: 2pt, fill: luma(250))[
          *Block B (Text)*\ $RR^128$\ #v(4pt) ↓\ #text(size: 9pt)[ProjHead]\ #v(4pt) ↓\ $RR^64$
        ],
        box(stroke: 0.5pt, inset: 8pt, radius: 2pt, fill: luma(250))[
          *Block C (Graph)*\ $RR^65$\ #v(4pt) ↓\ #text(size: 9pt)[ProjHead]\ #v(4pt) ↓\ $RR^32$
        ],
      )
      #v(15pt)
      #box(stroke: 0.8pt, inset: 10pt, radius: 2pt, fill: luma(240))[
        *Concatenation Layer*\ $bold(z)_i = [bold(h)_F parallel bold(h)_T parallel bold(h)_G] in RR^160$
      ]
      #v(15pt)
      #grid(
        columns: (1fr, 1fr),
        gutter: 15pt,
        box(stroke: 0.5pt, inset: 8pt, radius: 2pt)[
          *MLP / XGB Regressor*\ Output: $\hat{y}_i \in \RR$
        ],
        box(stroke: 0.5pt, inset: 8pt, radius: 2pt)[
          *MLP / XGB Classifier*\ Output: $\hat{p}_i \in [0, 1]$
        ],
      )
    ]
  ),
  caption: [Multimodal late-fusion architecture. The raw concatenated features (249-dim) are projected into a 160-dimensional joint representation $bold(z)_i$ before entering the prediction heads.],
) <fig-multimodal-fusion>

#figure(
  table(
    columns: (2fr, 1.5fr, 1.5fr, 1.5fr, 2fr),
    align: (left, center, center, center, center),
    inset: 8pt,
    stroke: 0.5pt,
    fill: (x, y) => if y == 0 { luma(240) },
    table.header(
      [*Stage*], [*Block A (Fin)*], [*Block B (Text)*], [*Block C (Graph)*], [*Concatenated*],
    ),
    [Raw features], [56], [128 (64+64)], [65 (64+1)], [249],
    [After ProjectionHead], [64], [64], [32], [$bold(z)_i = 160$],
    [M3e (+ aux scalars)], [—], [—], [—], [261],
  ),
  caption: [Canonical feature dimensionality across the pipeline architecture.],
) <tbl-dimensionality>

The primary model is the *late-fusion tri-modal neural architecture* implemented in #code-inline("src/models/fusion.py"). The final modelling setup is unambiguous: a Multi-Layer Perceptron (MLP) serves as the primary fusion engine, receiving a 160-dimensional projected embedding vector $bold(z)_i$. In parallel, an XGBoost classifier is trained on the raw 249-dimensional concatenated features as a non-linear baseline to verify whether the neural projection layer captures or obscures predictive signal. For the headline results, the MLP fusion model is the primary architecture. Each active stream passes through its own #code-inline("ProjectionHead") (linear + ReLU), and the resulting embeddings are concatenated (@eq-fusion-concat):

$ bold(z)_i = [bold(h)_F parallel bold(h)_T parallel bold(h)_G] in RR^(d_F' + d_T' + d_G') $ <eq-fusion-concat>

where $d_F' = 64$, $d_T' = 64$, $d_G' = 32$ by default.  To rigorously
evaluate both the magnitude of synergy and the practical investment decision,
the architecture implements a *Dual-Evaluation Framework*.  The concatenated
feature vector $bold(z)_i$ serves as the input to two parallel, independent
training pipelines:

- *Regressor Pipeline (H3):* Includes a two-layer MLP with a *linear* output
  activation (and XGBRegressor), optimised via Mean Squared Error (MSE) to
  predict continuous CAR.  This tests structural variance explanations.

- *Classifier Pipeline (H1, H2):* Includes a two-layer MLP with a *sigmoid*
  output activation (and XGBClassifier), optimised via Binary Cross-Entropy
  (BCE) to predict the binary direction of CAR ($1$ if $"CAR" > 0$, $0$
  otherwise).  This tests practical deal advisory discrimination.

Both pipelines share the same upstream concatenated representation $bold(z)_i$,
meaning the feature streams are extracted once and evaluated across both
objectives.

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
          #align(center)[*Stream A*\ Financial (Yahoo)\ $bold(h)_F in RR^56$]
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
      #v(6pt)
      #grid(
        columns: (1fr, 0.05fr, 1fr),
        gutter: 6pt,
        box(stroke: 0.4pt, inset: 6pt, radius: 3pt, width: 100%, fill: luma(245))[
          #align(center)[*Regressor Pipeline (H3)*\ #raw("train_models.py")\ MLP + XGBRegressor @chen2016 + Linear activation\ $hat(y)_i in RR$ (CAR magnitude)\ Loss: MSE]
        ],
        align(center + horizon)[],
        box(stroke: 0.4pt, inset: 6pt, radius: 3pt, width: 100%, fill: luma(245))[
          #align(center)[*Classifier Pipeline (H1, H2)*\ #raw("train_classifier.py")\ MLP + XGBClassifier @chen2016 + Sigmoid activation\ $hat(p)_i in [0,1]$ (CAR direction)\ Loss: BCE]
        ],
      )

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
- *Loss:* Two objectives trained in parallel: MSE for the Regressor Pipeline
  (continuous CAR); Binary Cross-Entropy for the Classifier Pipeline (binary CAR
  direction).  Huber loss sensitivity analysis is reported alongside MSE.
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
@mackinlay1997 @fama1973.  This section describes the full two-stage pipeline:
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
            #align(center)[#text(weight: "bold")[Stage 1 - OLS Event Study]\ #text(size: 8pt)[(#raw("compute_car.py"))]]
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
            #align(center)[#text(weight: "bold")[Stage 2 - Supervised Prediction]\ #text(size: 8pt)[(#raw("training_utils.py"))]]
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
            Losses (training):\\\
            $quad cal(L)_"MSE" = 1/N sum_i (y_i - hat(y)_i)^2 + lambda ||theta||_2^2$ (Regressor)\\\
            $quad cal(L)_"BCE" = -1/N sum_i [y_{"bin",i} log hat(p)_i + (1-y_{"bin",i}) log(1-hat(p)_i)] + lambda ||theta||_2^2$ (Classifier)\\\
            \\\
            Held-out metrics:\\\
            $quad$ MAE, RMSE, $R^2$ (regression); AUC-ROC, F1 (classification)
          ]
        ),
      )
      #v(8pt)
      #text(size: 8pt, style: "italic")[
        Stage 1 outputs are fixed market-derived labels independent of any model.
        Stage 2 uses pre-announcement features only - no post-deal information enters $bold(z)_i$.
      ]
    ]
  ),
  caption: [The two-stage CAR pipeline: Stage 1 computes actual CAR via OLS event study; Stage 2 trains and evaluates the fusion model against those labels.],
) <fig-car-pipeline>

=== Stage 1: OLS Market Model <sec-ols>

The market model is estimated over the *estimation window* $[-200, -20]$
trading days (180-day window, minimum 120 valid observations) using OLS
as implemented in #code-inline("scripts/data/compute_car.py") via
#code-inline("scipy.stats.linregress") as defined in @eq-market-model:

$ R_(i t) = alpha_i + beta_i R_(m t) + epsilon_(i t) $ <eq-market-model>

where:
- $R_(i t) = ln(P_(i t) \/ P_(i,t-1))$ is the acquirer log return on trading day $t$,
- $R_(m t)$ is the S&P 500 (SPX) log return on the same day,
- $hat(alpha)_i$ is the estimated intercept (abnormal return in absence of market movement),
- $hat(beta)_i$ is the estimated systematic risk loading (market beta) @sharpe1964.

The OLS estimators are:

$ hat(beta)_i = ("Cov"(R_i, R_m)) / ("Var"(R_m)) $ <eq-ols-beta>

$ hat(alpha)_i = overline(R)_i - hat(beta)_i overline(R)_m $ <eq-ols-alpha>

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

=== Abnormal Returns and CAR <sec-ar-car>

With $hat(alpha)_i$ and $hat(beta)_i$ estimated on the estimation window,
*abnormal returns* in the event window $cal(T) = {-5, ..., +5}$ are the
residuals between actual and model-predicted returns (@eq-ar):

$ A R_(i t) = R_(i t) - (hat(alpha)_i + hat(beta)_i R_(m t)) $ <eq-ar>

$A R_(i t)$ represents the return attributable to deal-specific information
(announcement effect) after stripping out normal market co-movement.
*CAR is the cumulative sum of these residuals* over the full event window (@eq-car-sum):

$ "CAR"_i = sum_(t=-5)^(+5) A R_(i t) $ <eq-car-sum>

#figure(
  image("../../docs/figures/fig_car_distribution.png", width: 90%),
  caption: [Distribution of deal-level $"CAR"_((-5,+5))$ across the full deal universe ($n approx 2,860$). The distribution is approximately symmetric with a slight negative mean, confirming the EMH-consistent noise dominance that renders continuous magnitude prediction structurally intractable (@tbl-reg-summary, $R^2 < 0$ across all configurations). The binary classification target ($"CAR" > 0$) partitions the distribution at zero --- the economically meaningful threshold.],
) <fig-car-distribution>

The near-symmetric noise distribution visible in @fig-car-distribution directly motivates the dual-pipeline evaluation design. Predicting the sign of CAR (classification) is tractable because deal fundamentals shift the mean; predicting the magnitude (regression) is intractable because the variance is dominated by unobservable announcement-day surprise --- a result confirmed empirically in @ch-findings.

This produces a single scalar per deal stored as column
#code-inline("car_m5_p5") in #code-inline("data/processed/final_car_dataset.csv").
A positive CAR indicates the market rewarded the acquisition announcement;
a negative CAR indicates value destruction.

=== Stage 2: Model Prediction vs. Actual CAR <sec-prediction-vs-actual>

The column #code-inline("car_m5_p5") (set as #code-inline("TARGET_COL") in
#code-inline("scripts/training/training_utils.py")) is the sole prediction
target for all model variants.  The fusion architecture minimises two distinct objective functions depending on
the evaluation pipeline:

*1. Continuous Target (Regression Pipeline --- H3):*
For variance analysis, the Regressor Pipeline minimises Mean Squared Error against
the raw continuous CAR ($y_i = "CAR"_i in RR$):

$ cal(L)_"MSE"(theta) = 1/N sum_(i=1)^N (y_i - hat(y)_i)^2 + lambda ||theta||_2^2 $ <eq-mse-loss>

*2. Binary Target (Classification Pipeline --- H1, H2):*
For directional discrimination, the Classifier Pipeline minimises Binary
Cross-Entropy against the thresholded label
$y_{"bin",i} = bb(1)["CAR"_i > 0] in {0, 1}$:

$ cal(L)_"BCE" (theta) = -1/N sum_(i=1)^N [y_{"bin",i} log hat(p)_i + (1 - y_{"bin",i}) log(1 - hat(p)_i)] + lambda ||theta||_2^2 $ <eq-bce-loss>

where $hat(p)_i = sigma(f_theta(bold(z)_i))$ is the sigmoid-activated synergy
probability.  All evaluation metrics are then computed by comparing predictions
against $y_i$ (regression) or $y_{"bin",i}$ (classification) on the held-out
test set using a strict chronological holdout (train: 2000–2016, val: 2017–2019, test: 2020–2023), with purged walk-forward cross-validation used within the training window only for hyperparameter selection, and an 11-day event-window embargo applied at each boundary @lopezdeprado2018:

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
    [$frac(1,N) sum_i |y_i - hat(y)_i|$ @willmott2005],
    [Primary metric; interpretable in percentage-point CAR terms.],

    [RMSE],
    [$sqrt(frac(1,N) sum_i (y_i - hat(y)_i)^2)$ @willmott2005],
    [Penalises large mispredictions more than MAE.],

    [$R^2$],
    [$1 - (sum_(i) (y_i - hat(y)_i)^2) / (sum_(i) (y_i - overline(y))^2)$],
    [Proportion of CAR variance explained by the model.],

    [Huber],
    [$sum_i cal(H)_delta (y_i - hat(y)_i)$ @huber1964],
    [Robust to outlier returns; sensitivity analysis only.],

    [Dir. Accuracy],
    [$frac(1,N) sum_i bb(1)[text("sign")(hat(y)_i) = text("sign")(y_i)]$ @pesaran1992 @leitch1991],
    [*Classification Pipeline.* Did the model predict the deal direction correctly?],

    [AUC-ROC],
    [$integral_0^1 "TPR"("FPR") thin d("FPR")$ @hanley1982],
    [*Primary classification metric.* Threshold-invariant measure of the model's
     ability to rank value-creating deals ($"CAR">0$) above value-destroying ones.],

    [F1-Score],
    [$2 times frac("Precision" times "Recall", "Precision" + "Recall")$ @vanrijsbergen1979],
    [Harmonic mean of precision and recall. Evaluates robustness under class
     imbalance in the CAR-positive vs. CAR-negative split.],
  ),
  caption: [Evaluation metrics. Regression metrics (MAE, RMSE, $R^2$, Huber) apply to the Regressor Pipeline; classification metrics (Dir. Accuracy, AUC-ROC, F1) apply to the Classifier Pipeline.],
) <tbl-metrics>

The primary design objective is that *no feature in $bold(z)_i$ is derived
from post-announcement data*.  Financial ratios $bold(h)_F$ use the most recent
pre-announcement fiscal year; FinBERT embeddings $bold(h)_T$ use the 10-K filed
before announcement; graph embeddings $bold(h)_G$ use SPLC relationships. While
the graph is structurally independent of deal outcomes, the static nature of SPLC
snapshots for early deals remains a methodological boundary condition as discussed
in @sec-block-c. The model therefore predicts market reaction from pre-deal information 
under these stated data constraints.

== Hypothesis Testing <sec-htesting>

Each hypothesis is tested through model ablation combined with statistical
significance testing:

- *H1 (Topological Alpha):* Compare M3 (#code-inline("full_fusion.yaml"))
  vs. M1 (#code-inline("financial_only.yaml")) on the *Classifier Pipeline*.
  While this comparison formally evaluates the joint contribution of both text and graph modalities against the financial baseline, the M2 ablation isolates the text effect, confirming that any residual lift in M3 is driven by topological structure. A paired $t$-test across cross-validation folds on held-out AUC-ROC scores
  assesses whether this full multimodal architecture yields a statistically significant
  improvement in directional discrimination ($p < 0.05$).

- *H2 (Semantic Divergence):* H2 is evaluated across two conditions:
  (a) a statistically significant Pearson/Spearman correlation between
  #code-inline("SemanticDiv_i") and $"CAR"_i$ ($p < 0.05$, primary test); and
  (b) M2 (#code-inline("financial_text.yaml")) yields an
  AUC-ROC improvement over M1 (#code-inline("financial_only.yaml")).
  If (b) fails, the mechanism is further evaluated to see if conflation is the cause.

- *H3 (Topological Arbitrage):* Levene's test for equality of variances is computed
  across betweenness centrality quantile groups to evaluate structural variance
  compression of announcement returns.

All tests use a significance threshold of $alpha = 0.05$ with *Bonferroni
correction* applied across the three hypothesis tests to control the
family-wise error rate ($alpha_"corrected" = 0.0167$).

== Evaluation Metrics <sec-metrics>

Evaluation metrics are partitioned by prediction head, as formalised in
@tbl-metrics.  The *Regressor Pipeline* (H3) is evaluated on continuous CAR
predictions using MAE (primary), RMSE, $R^2$, and Huber loss (sensitivity).
MAE is the primary regression metric given its interpretability in
percentage-point CAR terms and its robustness to the outlier return
distribution.

The *Classifier Pipeline* (H1, H2) is evaluated on binary CAR-direction
predictions using AUC-ROC (primary), F1-Score, and Directional Accuracy.
AUC-ROC is the primary classification metric because it is threshold-invariant
and directly measures the model's ability to rank value-creating deals above
value-destroying ones --- the economically actionable output for deal advisory.
F1-Score is reported as a secondary metric to assess robustness under class
imbalance in the CAR-positive vs. CAR-negative split.

== Ethical, Legal, and Social Considerations <sec-ethics>

While all data utilised in this study is sourced from public and institutional APIs (Yahoo Finance, Bloomberg) and contains no personally identifiable human-subject information, the deployment of machine learning in M&A strategy carries substantial socio-economic and ethical implications. The project was conducted under the terms of the ethical approval documentation provided in @appendix-ethics.

From a socio-economic perspective, models that successfully predict and therefore facilitate value-creating M&A transactions can accelerate corporate consolidation. Because the "synergies" realised in post-merger integration are frequently achieved through workforce redundancies and the elimination of overlapping operational departments, hyper-efficient AI-driven M&A targeting inherently risks accelerating job displacement. Furthermore, the concentration of capital enabled by algorithmically precise consolidation raises anti-trust and market-diversity concerns. 

Legally and professionally, the deployment of complex multimodal architectures (particularly Graph Neural Networks) in financial advisory introduces severe "black box" risks. Capital allocation decisions driven by opaque algorithmic processes violate the fiduciary duty of explicability required in corporate finance. This project explicitly mitigates this ethical and professional hazard by rejecting an end-to-end black box design in favour of a late-fusion architecture paired with SHAP decomposition (@lundberg2017). This ensures that every prediction is mathematically traceable back to its topological or semantic source, preserving the human advisor's ability to audit and justify the capital decision.

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

== Implementation and Auditability <sec-auditability>

To ensure the reproducibility of the study, the core modelling logic is maintained in a structured repository. Key implementation blocks—including the event-study CAR generation, temporal partitioning with embargo control, and the HeteroGraphSAGE architecture—are reproduced in @appendix-code. These snippets provide a readable and auditable record of the methodological safeguards described throughout this chapter.

== Summary

By standardising three disparate feature blocks within a rigorous dual-pipeline evaluation framework, the methodology operationalises the multi-modality requirement identified in the literature. The following chapter reports the empirical results of this architecture, testing whether the theoretical advantages of topological and textual fusion translate into measurable predictive alpha.
