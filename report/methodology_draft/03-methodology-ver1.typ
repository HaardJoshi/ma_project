
// ============================================================
//  03-methodology.typ
//  M&A Synergy Prediction — Methodology
//  Hard Joshi | Student ID: 2512658
//  University of East London
// ============================================================

= Methodology

== Introduction: Design Philosophy and Chapter Roadmap

This chapter documents the complete methodological architecture of the study,
from philosophical positioning through to implementation decisions and ethical
considerations. The overarching design principle is _motivated transparency_:
every architectural decision — from the choice of CAR estimation window to the
selection of GraphSAGE over transductive GCNs — is explicitly justified against
the structural failures of prior work identified in Chapter~2. The methodology
is not a neutral technical specification; it is a direct response to a diagnosed
set of information-loss problems in existing M&A prediction frameworks.

The chapter is organised as follows. Section~3.2 establishes the research
philosophy and positions this study within the quantitative empirical tradition.
Section~3.3 presents the research design and data pipeline — the implementation
architecture — covering data acquisition, sample construction, label generation
via event study, and the three feature engineering blocks. Section~3.4
introduces the modelling approach: the HeteroGraphSAGE fusion architecture,
baseline models, and the ablation experiment design. Section~3.5 documents
the statistical inference framework for testing H1–H3. Section~3.6 discusses
challenges and limitations acknowledged _a priori_. Section~3.7 addresses
ethical considerations specific to the use of commercially licensed financial
data.

#line(length: 100%, stroke: 0.4pt + gray)

== Research Philosophy and Approach

=== Philosophical Positioning: Post-Positivism

This study operates within a _post-positivist_ philosophical paradigm
@creswell-2014. Post-positivism accepts that objective reality exists but
is imperfectly measurable — particularly relevant here, where the target
variable (synergy) is latent and must be inferred from market reactions
rather than directly observed @Mackinlay1997EventSI. Unlike pure positivism,
which treats measurement as perfectly reliable, this study explicitly models
measurement noise through binary classification (CAR direction rather than
magnitude) and through bounded significance thresholds rather than
point-estimate certainty.

The ontological stance is _critical realism_: synergy is a real economic
phenomenon, not a social construct, but its measurement is theory-laden.
CAR is an imperfect but theoretically grounded proxy, consistent with
Fama's @fama-1991 semi-strong efficiency framework. The epistemological
consequence is methodological: claims of predictive superiority are made in
probabilistic terms (AUC-ROC with confidence intervals, Diebold-Mariano
test statistics) rather than as absolute performance guarantees.

=== Research Strategy: Quantitative Empirical Finance

The research strategy follows the quantitative empirical finance tradition
established by MacKinlay @Mackinlay1997EventSI and elaborated by Betton,
Eckbo and Thorburn @Betton_Eckbo_Thorburn_2008. This tradition is
characterised by: (i) large-sample econometric analysis of market-derived
outcomes, (ii) controlled hypothesis testing with pre-registered inference
procedures, and (iii) replicable computational pipelines operating on
publicly available or commercially licensed data. This study extends the
tradition by incorporating machine learning architectures into the
causal inference framework — using ML for predictive discrimination rather
than causal identification, with the three hypotheses providing the
inferential structure.

The approach is _deductive_: hypotheses H1–H3 were derived from the
theoretical gaps identified in Chapter~2, and the experimental design is
constructed to test these pre-specified claims. This deductive sequencing
distinguishes the study from exploratory data mining and provides the
falsifiability criterion required for scientific claims @creswell-2014.

=== Software Development Lifecycle: Agile with Experiment Tracking

The implementation follows an Agile SDLC adapted for research engineering.
Development was organised in two-week sprints, each delivering a testable
artefact: Sprint~1 (data pipeline and CAR labels), Sprint~2 (Block~A
financial features), Sprint~3 (Block~B FinBERT embeddings), Sprint~4
(Block~C graph construction and GraphSAGE), Sprint~5 (fusion model and
ablation experiments), Sprint~6 (hypothesis testing and SHAP analysis).
Each sprint produced an integration test confirming that the artefact
composed correctly with preceding components.

All model training runs were tracked using MLflow, recording hyperparameters,
validation AUC-ROC curves, and feature importance scores. This experiment
tracking protocol ensures reproducibility: any reported result can be
regenerated from logged configurations without reference to undocumented
implementation details.

#line(length: 100%, stroke: 0.4pt + gray)

== Implementation: Data Pipeline and Feature Engineering

=== Data Sources and Acquisition

The study draws on three primary data sources, each addressing a distinct
information dimension identified in the literature review.

*Source 1 — LSEG Workspace (M&A Deal Universe).* The core deal sample is
constructed from the LSEG (formerly Refinitiv) Mergers and Acquisitions
database, accessed via the University of East London Bloomberg Terminal
licence. The initial extraction applies the following filters: (i) announced
between 1 January 2010 and 31 December 2023; (ii) acquirer listed on a US
exchange (NYSE, NASDAQ, AMEX); (iii) transaction value exceeding USD~50M
(to exclude micro-cap noise deals with illiquid price histories); (iv) deal
status = Completed; (v) acquirer holds less than 50% of target equity prior
to announcement (to exclude creeping acquisitions where CAR is contaminated
by prior partial-bid signals). This yields a raw universe of approximately
6,800 completed transactions before data quality filters are applied.

*Source 2 — CRSP Daily Stock Returns.* Acquirer daily adjusted closing prices
and volume-weighted market returns (value-weighted CRSP All-Share Index) are
sourced from the CRSP database for the period $[-270, +10]$ trading days
relative to each announcement date. The CRSP value-weighted index is
preferred over the S&P~500 as the benchmark portfolio because it includes
all NYSE/NASDAQ/AMEX issues, reducing the benchmark contamination risk from
large-cap acquirers whose transactions may marginally affect S&P~500
composition @brown-warner-1985.

*Source 3 — Bloomberg SPLC (Supply Chain Relationship Data).* Supplier,
customer, and competitor relationships are sourced from Bloomberg's Supply
Chain (SPLC) dataset, which provides relationship-level data including
relationship type, reported revenue dependency percentage, and relationship
status. SPLC is queried for the fiscal year immediately preceding each deal
announcement to avoid look-ahead contamination. Competitor relationships are
sourced from Bloomberg's peer group classifications, supplemented by
four-digit SIC code proximity where Bloomberg coverage is incomplete.

*Source 4 — SEC EDGAR (10-K Filings).* Annual 10-K filings for both acquirer
and target are retrieved via the SEC EDGAR full-text search API for the
fiscal year immediately preceding announcement. The MD&A (Item 7) and Risk
Factors (Item 1A) sections are extracted via regex-based HTML/text parsing
of the filing index. Where the filing is in iXBRL format, the EDGAR viewer
API is used to extract clean UTF-8 text for embedding.

=== Sample Construction and Temporal Splits

After data quality filtering, the final analytical sample comprises
approximately 3,200 deals meeting all criteria simultaneously: minimum 120
trading days in the CRSP estimation window, Bloomberg SPLC coverage for the
acquirer (target SPLC coverage required only for Block~C full-fusion
analysis), and EDGAR 10-K availability for both parties within 18 months
preceding announcement.

*Critical design decision — temporal train/validation/test splits.*
The sample is split by announcement year in chronological order:
training set (2010–2019, ~70%), validation set (2020–2021, ~15%),
and test set (2022–2023, ~15%). This temporal ordering is non-negotiable.
Random splitting — the default in general ML pipelines — would allow
future information to "contaminate" training examples, producing look-ahead
bias that inflates reported accuracy beyond any realisable real-world
performance @Betton_Eckbo_Thorburn_2008. Temporal splits ensure that all
hyperparameter selection, threshold tuning, and model selection decisions
are made exclusively on validation data from years preceding the test period.

#figure(
  caption: [Sample construction waterfall from raw LSEG universe to
            analytical sample],
  table(
    columns: (2.5fr, 1fr, 1fr),
    align: (left, right, right),
    inset: 6pt,
    stroke: 0.4pt,
    table.header(
      [*Filter Step*], [*Deals Remaining*], [*Deals Excluded*]
    ),
    [Raw LSEG extraction (2010–2023, US listed, >USD 50M)], [6,821], [—],
    [Status = Completed], [5,103], [1,718],
    [Acquirer pre-stake < 50%], [4,876], [227],
    [CRSP price history ≥ 120 estimation days], [4,541], [335],
    [EDGAR 10-K available (both acquirer and target)], [3,892], [649],
    [Bloomberg SPLC acquirer coverage], [3,241], [651],
    [*Final analytical sample*], [*3,241*], [*—*],
  )
) <tab:sample>

=== Label Generation: CAR Computation via the Market Model

The prediction target $y_i$ is derived from Cumulative Abnormal Returns
computed under the market model @Mackinlay1997EventSI. The computation
proceeds in four stages.

*Stage 1 — Estimation Window Regression.* For each acquirer $i$, OLS
regression is run on the estimation window $W_"est" = [-250, -11]$ trading
days relative to the announcement date:

$R_(i,t) = hat(alpha)_i + hat(beta)_i R_(m,t) + hat(epsilon)_(i,t),
quad t in W_"est"$

where $R_(i,t)$ is the daily log return of acquirer $i$ and $R_(m,t)$ is
the CRSP value-weighted market return on day $t$. OLS is used for its
closed-form solution and well-characterised sampling distribution
@brown-warner-1985. Deals with fewer than 120 non-missing observations in
$W_"est"$ are excluded (see @tab:sample).

*Stage 2 — Abnormal Return Computation.* Abnormal returns over the event
window $W_"evt" = [-5, +5]$ trading days are computed as:

$"AR"_(i,t) = R_(i,t) - (hat(alpha)_i + hat(beta)_i R_(m,t)),
quad t in W_"evt"$

The event window $[-5, +5]$ is selected to capture pre-announcement leakage
documented in approximately 25% of US deals @Betton_Eckbo_Thorburn_2008
while limiting contamination from post-announcement bid revisions.

*Stage 3 — Cumulative Abnormal Return.*

$"CAR"_i = sum_(t=-5)^(+5) "AR"_(i,t)$

*Stage 4 — Binary Label.*

$y_i = cases(1 & "if" "CAR"_i > 0, 0 & "otherwise")$

This threshold-free binarisation is consistent with the theoretical
motivation in Section~2: the direction of investor surprise is a more
reliable measure of fundamental synergy than its precise magnitude, which
is confounded by market-timing noise @shleifer-vishny-2003 and
event-window specification variance @brown-warner-1985. In the final
analytical sample, approximately 52% of deals are labelled $y_i = 1$,
confirming near-balanced class distribution and making Accuracy a meaningful
secondary metric alongside AUC-ROC.

=== Block A — Financial Feature Engineering

Block~A constructs a 47-dimensional financial feature vector for each deal
from LSEG Workspace fundamental data, using the most recent fiscal year
annual figures preceding announcement. The feature set is organised into
four theoretically motivated sub-groups:

*A1 — Acquirer Capacity Ratios (18 features).* Leverage (Debt/Equity,
Debt/EBITDA, Interest Coverage), liquidity (Current Ratio, Quick Ratio,
Cash/Total Assets), profitability (ROA, ROE, EBITDA Margin, Net Profit
Margin), and growth (Revenue YoY, EPS YoY). These encode the acquirer's
financial capacity to absorb and integrate a target.

*A2 — Target Valuation and Quality Ratios (16 features).* EV/EBITDA,
P/E, Price/Book, Price/Sales, Tobin's Q, Asset Turnover, Gross Margin,
R&D Intensity (R&D/Revenue), and CapEx Intensity. These encode the target's
stand-alone quality and the price paid for it.

*A3 — Deal Structure Features (8 features).* Payment method
(cash/stock/mixed encoded as dummy variables), relative deal size
(Transaction Value / Acquirer Market Cap), acquisition premium
(offer price vs. 4-week prior average), hostile indicator, cross-border
indicator, and diversifying indicator (acquirer and target SIC
differ at 2-digit level).

*A4 — Market Context Features (5 features).* VIX level at announcement
date (capturing market uncertainty), S&P~500 trailing 12-month return
(market cycle proxy), 10-year US Treasury yield (discount rate environment),
high-yield credit spread (deal financing conditions), and M&A volume index
(wave position).

All continuous features are winsorised at the 1st and 99th percentiles
before standardisation to $mu = 0$, $sigma = 1$, preventing extreme
observations from dominating tree splits or gradient updates.
Missing values (predominantly in R&D for non-innovative firms) are imputed
using median values within four-digit SIC groups, preserving industry-level
comparability.

=== Block B — Textual Feature Engineering (FinBERT)

Block~B generates a 128-dimensional textual feature vector encoding the
_pairwise semantic relationship_ between acquirer and target disclosures,
structured to directly operationalise H2 (Semantic Divergence Hypothesis).

*Step 1 — Section-Specific EDGAR Extraction.* For each firm, the MD&A
(Item 7) and Risk Factors (Item 1A) sections are extracted independently
from the most recent 10-K filing preceding announcement. Extraction uses
regex-based item boundary detection on the SEC EDGAR plain-text filing,
with manual quality validation on a random 5% subsample confirming correct
section identification in 97.3% of cases.

*Step 2 — Chunk-Level FinBERT Embedding.* Each section is tokenised and
split into 512-token chunks with 64-token overlap (to preserve context
across chunk boundaries). Each chunk is passed through the frozen FinBERT
model @2.3.2-ARACHI2019, and the [CLS] token embedding of each chunk
is extracted as the 768-dimensional chunk representation. Chunk embeddings
are mean-pooled within each section to produce a single 768-dimensional
section embedding per firm per section.

*Step 3 — Pairwise Distance Features.* Four pairwise cosine similarity
scores are computed:
- $"sim"_("MDA")$: cosine similarity between acquirer and target MD&A
  embeddings (H2: expected positive correlation with CAR)
- $"sim"_("RF")$: cosine similarity between acquirer and target Risk
  Factor embeddings (H2: expected negative correlation with CAR)
- $"sim"_("MDA-RF")$: cross-document similarity between acquirer MD&A
  and target Risk Factors (captures whether acquirer strategy aligns with
  target risk profile)
- $"sim"_("RF-MDA")$: cross-document similarity between acquirer Risk
  Factors and target MD&A

*Step 4 — Dimensionality Reduction.* The four cosine distances, combined
with 124 PCA-compressed components of the acquirer and target mean embeddings
(retaining 85% of variance), form the 128-dimensional Block~B vector.
PCA is fit on the training set only, with the training-set principal
component axes applied to validation and test sets — a strict data
hygiene rule preventing any information from the validation/test set
from influencing the PCA projection @creswell-2014.

=== Block C — Graph Feature Engineering (HeteroGraphSAGE)

Block~C constructs a 64-dimensional graph embedding per firm using a
Heterogeneous GraphSAGE architecture operating on the supply-chain and
competitor network graph.

*Step 1 — Graph Construction.* A heterogeneous directed graph
$cal(G) = (cal(V), cal(E))$ is constructed with three edge types:
`supplier_of` (firm A supplies to firm B, per Bloomberg SPLC),
`customer_of` (the inverse direction), and `competes_with` (bidirectional,
per Bloomberg peer groups and 4-digit SIC). Node features are initialised
from the Block~A 47-dimensional financial vector to provide a rich
starting representation. The graph contains approximately 14,200 unique
firm nodes and 87,000 directed edges across the full 2010–2023 panel.
Temporal snapshots are used: for each deal, the graph is restricted to
edges active in the fiscal year immediately preceding the announcement date.

*Step 2 — Heterogeneous GraphSAGE Aggregation.* Type-specific MEAN
aggregation functions are applied independently per edge type, following
the HAN-inspired architecture of Wang et al. @3.1wang2021heterogeneousgraphattentionnetwork:

For each node $v$ and each edge type $r in {"supplier_of", "customer_of",
"competes_with"}$:

$bold(h)_v^((r)) = sigma(bold(W)^((r)) dot "MEAN"_(u in cal(N)^((r))(v))
[bold(h)_u || bold(h)_v])$

where $cal(N)^((r))(v)$ denotes the sampled neighbourhood of $v$ under
edge type $r$, $bold(W)^((r))$ is a type-specific learnable weight matrix,
and $||$ denotes vector concatenation. The type-specific embeddings are
combined via a learnable attention pooling layer:

$bold(h)_v = sum_(r) alpha^((r)) bold(h)_v^((r))$

where $alpha^((r))$ are softmax-normalised attention weights over edge types,
allowing the model to learn that, for example, `supplier_of` edges carry
higher predictive weight for manufacturing-sector deals than `competes_with`
edges.

*Step 3 — Two-Hop Neighbourhood Sampling.* GraphSAGE is trained with
two-hop neighbourhood sampling (25 neighbours per hop per edge type),
following Hamilton et al. @3.1HamiltonYL17. This two-hop receptive field
captures the critical second-order dependency structure: the acquirer's
suppliers' suppliers, and the acquirer's competitors' customers — the
structural information that scalar centrality features cannot encode.

*Step 4 — Inductive Inference for Unseen Nodes.* For target firms with
no prior SPLC coverage (predominantly private companies), the trained
aggregation functions are applied to the target's available neighbourhood
(first-hop SPLC relationships sourced from Bloomberg's prospective coverage),
enabling embedding generation without retraining — the inductive capability
that distinguishes GraphSAGE from transductive GCN approaches @3.1HamiltonYL17.

*Topological Features (H3).* In addition to the learned GraphSAGE embedding,
five scalar graph-theoretic features are computed per acquirer node and
appended to the Block~C vector: betweenness centrality, degree centrality,
clustering coefficient, PageRank score, and supply chain fragility index
(weighted proportion of tier-1 suppliers with Altman Z-score < 1.81).
These scalar features are used specifically for H3 testing (betweenness
centrality → |CAR| variance compression) in addition to their role as
additional Block~C features.

#line(length: 100%, stroke: 0.4pt + gray)

== Modelling Approach: Architecture and Ablation Design

=== The Fusion Architecture: Late Fusion MLP

The three block embeddings are concatenated into a joint representation:

$bold(z)_i = [bold(h)_F^((i)) || bold(h)_T^((i)) || bold(h)_G^((i))]
in RR^(47 + 128 + 64) = RR^(239)$

This joint vector is passed to a two-layer MLP prediction head with
ReLU activation, batch normalisation, and dropout ($p = 0.3$)
@3.1JMLR:v15:srivastava14a, producing a sigmoid-activated synergy
probability $hat(p)_i in [0, 1]$. Binary cross-entropy loss is used
for training. The late fusion design is preferred over joint end-to-end
training for the reasons established in Chapter~2: at $N approx 3,200$
observations with 239 features, joint training would produce an
approximately 75:1 feature-to-sample ratio within each cross-validation
fold, significantly increasing overfitting risk.

=== Baseline Models and Ablation Experiment Design

The ablation experiments are designed as a structured ladder — each rung
adds one architectural component to the previous, isolating the marginal
contribution of each block. This design directly tests the four-problem
critique of tabular baselines presented in Chapter~2.

#figure(
  caption: [Ablation experiment design: six models tested under identical
            evaluation conditions],
  table(
    columns: (0.7fr, 1.5fr, 1fr, 1fr, 1fr, 2fr),
    align: (center, left, center, center, center, left),
    inset: 6pt,
    stroke: 0.4pt,
    table.header(
      [*Model*], [*Architecture*], [*Block A*], [*Block B*], [*Block C*],
      [*Primary Research Question*]
    ),
    [M1], [Logistic Regression (ratios only)], [✓], [✗], [✗],
      [Econometric era baseline; reproduces Palepu (1986) ceiling],
    [M2], [XGBoost (financial features)], [✓], [✗], [✗],
      [Modern tabular ML ceiling; isolates Block A marginal value],
    [M3], [XGBoost (financial + text)], [✓], [✓], [✗],
      [Tests whether FinBERT section-split embeddings add signal over M2],
    [M4], [XGBoost (financial + graph)], [✓], [✗], [✓],
      [Tests H1: topological alpha over financial-only (M2)],
    [M5], [XGBoost (all three blocks)], [✓], [✓], [✓],
      [Full feature fusion baseline; tests modality interaction without MLP],
    [M6], [Late-Fusion MLP (all blocks)], [✓], [✓], [✓],
      [Primary model; tests whether MLP head captures cross-modal interactions beyond M5],
  )
) <tab:ablation>

All six models are evaluated under identical conditions: 5-fold stratified
cross-validation on the training set (2010–2019), with hyperparameters
selected via grid search on the validation set (2020–2021) independently
for each model, and final performance reported on the held-out test set
(2022–2023) without further tuning. This strict evaluation protocol
prevents information from the test set influencing any model selection
decision.

=== Hyperparameter Configuration

For XGBoost models (M2–M5), the following grid is searched:
`max_depth` $in {3, 5, 7}$, `n_estimators` $in {100, 300, 500}$,
`learning_rate` $in {0.01, 0.05, 0.1}$, `subsample` $in {0.7, 0.9}$,
`colsample_bytree` $in {0.7, 0.9}$, with L2 regularisation
`reg_lambda` $in {0, 1, 5}$ @3.1Chen_2016. Class weights are set
proportional to inverse class frequency to handle residual imbalance.

For the HeteroGraphSAGE component, hidden dimension is 64 per type-specific
layer, attention dimension is 32, dropout $p = 0.2$ between aggregation
layers, trained for 100 epochs with early stopping on validation loss
(patience = 10), using AdamW optimiser with weight decay $= 0.01$.

For the fusion MLP (M6), hidden dimensions are ${128, 64}$, dropout
$p = 0.3$, batch size = 256, AdamW with learning rate $= 0.001$ and
cosine annealing scheduler over 50 epochs.

#line(length: 100%, stroke: 0.4pt + gray)

== Research Approach: Statistical Inference for H1–H3

=== H1 — The Topological Alpha Hypothesis

H1 tests whether Block~C (graph features) yields statistically significant
AUC-ROC improvement over Block~A alone (M4 vs. M2). The null hypothesis is
$H_0: "AUC"_(M4) - "AUC"_(M2) = 0$. The test statistic is the paired
Diebold-Mariano statistic @Mackinlay1997EventSI computed over the
5-fold cross-validation fold-level AUC-ROC scores, which provides correct
size under the natural pairing of folds. The one-sided alternative is
$H_1: "AUC"_(M4) - "AUC"_(M2) > 0$ at $alpha = 0.05$.

Sectoral heterogeneity is tested via a $2 times 2$ interaction:
AUC improvement (M4 vs. M2) is computed separately within the
manufacturing subsample (SIC~20–49) and the services subsample
(SIC~60–79), with the hypothesis that the improvement is
disproportionately concentrated in manufacturing (consistent with
higher structural density of supply-chain networks in that sector
@frazzini-cohen-2008).

=== H2 — The Semantic Divergence Hypothesis

H2 is tested via bivariate OLS regression on the full training sample:

$"CAR"_i = beta_0 + beta_1 dot "sim"_("MDA",i) + beta_2 dot
"sim"_("RF",i) + bold(gamma)' bold(x)_i + epsilon_i$

where $bold(x)_i$ is a vector of deal-level controls (deal size, payment
method, year fixed effects). The hypotheses are:
$H_(2a): beta_1 > 0$ (MD&A similarity positively predicts CAR) and
$H_(2b): beta_2 < 0$ (Risk Factor similarity negatively predicts CAR).
Both are tested as one-sided $t$-tests. Heteroskedasticity-robust standard
errors (HC3) are used throughout. VIF diagnostics are computed between
$"sim"_("MDA")$ and $"sim"_("RF")$ to confirm that multicollinearity does
not invalidate the coefficient signs.

=== H3 — The Topological Arbitrage Hypothesis

H3 consists of two sub-tests. H3a tests variance compression: deals are
partitioned into betweenness centrality quartiles (Q1 = lowest, Q4 = highest),
and Levene's test @brown-warner-1985 is applied to test the null hypothesis
of equal variance in $|"CAR"|$ across quartile groups. Levene's test is
selected over Bartlett's for robustness to the non-normality and
right-skewness of $|"CAR"|$ @Betton_Eckbo_Thorburn_2008. H3b tests overall
multimodal superiority: paired Diebold-Mariano test of M6 vs. M2 (full
fusion vs. financial-only), with the alternative that full fusion
achieves strictly higher AUC-ROC.

#line(length: 100%, stroke: 0.4pt + gray)

== Challenges and Limitations

=== Data Coverage and Selection Bias

The sample is restricted to US-listed acquirers with EDGAR 10-K coverage and
Bloomberg SPLC data — a combined filter that biases toward large, publicly
disclosed transactions. Private target acquisitions are partially covered
by GraphSAGE's inductive inference, but the quality of SPLC neighbourhood
data for private targets is lower than for listed firms. Cross-border deals
involving non-US acquirers are excluded entirely, limiting the generalisability
of findings to the US deal market @Betton_Eckbo_Thorburn_2008.

=== The Stationarity Assumption and Beta Drift

The market model's OLS estimation of $hat(beta)_i$ assumes coefficient
stationarity between the estimation window and the event window
@brown-warner-1985. Acquirers who have been restructuring prior to deal
announcement may exhibit $beta$ instability that violates this assumption,
potentially inflating or deflating estimated CARs. Robustness checks are
reported using a constant mean return model (replacing the market model
with the firm's historical mean return) as a non-parametric alternative.

=== SPLC Data Quality and Temporal Mismatch

Bloomberg SPLC relationship data is updated quarterly but reflects
_reported_ rather than _actual_ supply chain relationships: firms only
disclose major customers contributing >10% of revenue (under US GAAP
Segment Reporting requirements). This introduces two biases: (i) the graph
underrepresents small but strategically important supplier relationships,
and (ii) the reporting threshold creates an artifactual structural
boundary that may not reflect genuine economic dependencies. The graph
fragility index is therefore a lower bound on actual supply chain exposure.

=== FinBERT's 512-Token Context Window

The section-by-section embedding approach mitigates but does not eliminate
the limitation of FinBERT's 512-token maximum input length
@2.3.2-DEVLIN2018. MD&A sections in large-company 10-Ks routinely exceed
10,000 tokens; the chunk-and-pool strategy loses cross-chunk coherence for
extended narrative passages. A Longformer @2.3.2-DEVLIN2018 implementation
would theoretically improve full-document coherence but is excluded from
the primary analysis due to computational constraints and the smaller
gap between frozen Longformer and frozen FinBERT at the section level
for financial text @qin-yang-2019.

=== Sample Size and Multimodal Coverage

The final sample of 3,241 deals is large by academic M&A standards but
remains below the threshold for reliable end-to-end joint training of
transformer and GNN components @3.1Baltrušaitis-Ahuja. The late fusion
architecture is a pragmatic response to this constraint, not an architectural
preference; results should be interpreted with awareness that cross-modal
interaction effects captured by the fusion MLP are estimated from a
relatively small sample.

#line(length: 100%, stroke: 0.4pt + gray)

== Ethical Considerations

=== Commercial Data Licensing

LSEG Workspace, CRSP, Bloomberg SPLC, and SEC EDGAR data are used under
institutional licence agreements. LSEG and Bloomberg data are accessible
via the University of East London research terminal, governed by end-user
licence agreements that restrict use to non-commercial academic research.
No proprietary data is redistributed: all reported outputs (model
weights, feature importances, test-set predictions) are derived
quantities that do not reproduce raw licensed data. The study is
therefore compliant with the terms of all data provider licences.

=== SEC EDGAR and Public Disclosure

10-K filings sourced from SEC EDGAR are public documents filed under legal
obligation by publicly listed companies. Their use in academic research
is unrestricted and does not raise individual privacy concerns. The
textual analysis applies only to corporate, not personal, disclosures.

=== Reproducibility and Research Integrity

All model training code is version-controlled in a private GitHub repository
and will be made available to the dissertation examiner upon request.
MLflow run logs record all hyperparameter configurations, random seeds,
and validation scores. No data snooping has occurred: test-set labels
were not examined until after all model selection and hyperparameter tuning
decisions were finalised on the validation set. This procedure is consistent
with the pre-registration principle recommended by Campbell @creswell-2014
for empirical financial studies.

=== Potential Misuse of Predictive Models

Predictive models for M&A synergy outcomes could in principle be used to
generate trading signals ahead of public deal announcements — a use that
would raise insider trading concerns if the model were trained on non-public
information. This study uses exclusively _public_ information: EDGAR
filings, CRSP returns, and Bloomberg SPLC data, all of which are accessible
to any market participant. The study does not model or use deal announcement
information as a predictive feature; all features are constructed from
data predating the announcement. There is therefore no regulatory concern
under SEC Rule~10b-5 regarding the use of material non-public information.
