
// ============================================================
//  02-lit-review.typ  (v2 — final polished submission draft)
//  M&A Synergy Prediction — Literature Review
//  Hard Joshi | Student ID: 2512658
//  University of East London
// ============================================================

= Literature Review

== The Central Argument: A Gap Three Decades Wide

Existing quantitative frameworks for predicting post-acquisition synergy are
fundamentally constrained by a shared architectural assumption: they treat the
firm as an isolated data point, devoid of context, history, or structural
position. Financial ratio models, machine-learning classifiers, and even
transformer-based NLP pipelines all process firms as nodes without edges —
objects detached from the industrial ecosystems that define their actual value.
This literature review builds a sustained critical argument across four
interconnected knowledge streams:

+ *Stream I — The M&A Paradox and the Failure of Valuation Theory*, which
  diagnoses _why_ synergy prediction is difficult and demonstrates that the
  root cause is structural information loss, not computational insufficiency.

+ *Stream II — The Tabular Paradigm*, covering the full history of
  quantitative M&A models from logistic regression through gradient boosting
  and early deep learning, demonstrating that each generation inherits the
  same fatal independence assumption — and establishing the precise ceiling
  beyond which single-modality tabular models cannot advance.

+ *Stream III — The Semantic Turn*, tracing the evolution of financial NLP
  from bag-of-words lexicons to transformer architectures, critically
  demonstrating that despite increasing sophistication, the field systematically
  misdirects powerful tools at deal _occurrence_ rather than deal _outcome_,
  and that even FinBERT implementations fall into a "Tabular Trap" the moment
  their embeddings are flattened into scalar vectors.

+ *Stream IV — The Topological Turn*, engaging with the supply chain finance,
  corporate networks, and graph neural network literatures to establish that
  network topology carries _irreducible_ economic signal about synergy
  potential — signal that is mathematically unrecoverable by any non-graph
  model regardless of its parameter count.

The review concludes that these four streams converge on a single logical
exit strategy: a heterogeneous graph architecture that simultaneously fuses
financial fundamentals (Block~A), domain-specific textual embeddings
(Block~B), and supply-chain/competitor graph topology (Block~C) within a
unified predictive framework — a configuration that, as of 2025, no published
study has deployed for post-merger synergy _outcome_ prediction. Every section
below is therefore not a neutral survey but a _motivated critique_: prior work
is evaluated for what it structurally _cannot_ achieve, and each limitation is
mapped directly to one of the three hypotheses tested in this dissertation.

#line(length: 100%, stroke: 0.4pt + gray)

== Stream I — The M&A Paradox and the Failure of Valuation Theory

=== The Empirical Record: Stationarity of Failure

Mergers and acquisitions remain the primary mechanism for global capital
reallocation, yet their empirical track record constitutes one of the most
durable anomalies in corporate finance. Martynova and Renneboog
@MARTYNOVA20082148 documented that between 70% and 90% of acquisitions fail
to generate value for acquirer shareholders across a full century of global
takeover activity — a finding replicated across multiple independent
meta-analyses @Christensen_Alton_Rising_Waldeck_2011. Critically, this
failure rate has exhibited _statistical stationarity_ across distinct economic
regimes — from the conglomerate wave of the 1960s through the dot-com era and
post-financial-crisis consolidations — implying that the root cause is
structural rather than cyclical @MARTYNOVA20082148.

Bradley, Desai, and Kim @BRADLEY19883 established that while target
shareholders captured average announcement-date abnormal returns of
approximately 30%, acquirer shareholders experienced systematic losses,
yielding combined synergy gains averaging only 7.4% of combined pre-deal
firm value. This gap between the value created for targets and the value
transferred (or destroyed) for acquirers is the precise economic surplus that
a reliable predictive model should identify _ex ante_. The persistence of
this destruction in the face of progressively more sophisticated
computational due diligence tools motivates the epistemological question this
study addresses: what information, structurally absent from conventional
models, would close this predictive gap?

=== Behavioural Distortions: Hubris as Label Noise

The first explanatory layer is behavioural. Roll's Hubris Hypothesis
@Roll1986TheHH argues that acquiring managers systematically overestimate
their ability to extract value, treating the target's efficient market price
as an underestimate rather than an unbiased signal. This creates the "Synergy
Trap" identified by Sirower @sirower_synergy_1997, wherein the premium paid
mathematically requires performance improvements that are unattainable in the
majority of cases.

From a machine learning perspective, hubris introduces systematic _label
noise_ into training datasets: if acquisition premiums partly reflect
executive overconfidence rather than genuine synergy potential
@Roll1986TheHH, models trained on transaction values are fitting a signal
contaminated with behavioural error. This contamination is _irreducible_ —
no amount of regularisation or hyperparameter tuning can remove bias that is
encoded into the target variable itself. This motivates the use of
Cumulative Abnormal Return (CAR) as the prediction target rather than deal
premia or accounting synergies, since CAR represents the market's _ex post_
revision of expectations and is less susceptible to managerial narrative.

=== Market Timing and the Stock-Driven Wave

A complementary source of label contamination is identified by Shleifer and
Vishny @shleifer-vishny-2003, whose stock-market-driven acquisitions model
demonstrates that overvalued acquirers rationally exploit inflated equity as
acquisition currency during market peaks. Under this framework, a substantial
proportion of M&A waves are motivated by relative misvaluation rather than
operational complementarity — the observed CAR reflects market correction
dynamics rather than synergy realisation. Grossman and Stiglitz
@grossman-stiglitz-1980 established the theoretical bound here: in a world
of costly information acquisition, markets cannot be fully informationally
efficient, and persistent information asymmetries around deal announcements
are a natural equilibrium. This "bounded efficiency" framing justifies the
premise that pre-announcement features can contain predictive signal about
CAR direction — information costs prevent instantaneous arbitrage of the
signal this study attempts to extract.

=== The Epistemological Barrier: Structural Blindness

Even setting aside behavioural distortions, the verification mechanism is
structurally deficient. Angwin @angwin_mergers_2001 documented that
conventional due diligence operates in organisational silos — financial,
legal, and operational teams independently assess their respective domains
without modelling interaction effects. This orthogonal approach misses
multiplicative risks: strong cash flows combined with a fragile single-source
supplier dependency produce a combined exposure that sums to near-zero value;
linear due diligence passes both in isolation and misses the conjunction.

Akerlof's @akerlof-1970 "Market for Lemons" framework provides the
theoretical grounding for this structural blindness: mandated disclosures
(10-K filings) are insufficient to eliminate information asymmetry because
targets can strategically obscure ecosystem vulnerabilities within the noise
of standard reporting formats @Hansen1987. The implication is precise:
reliable synergy estimation requires information sources that cannot be
strategically manipulated — specifically, the verifiable topology of the
firm's external network of suppliers, customers, and competitors. This forms
the primary theoretical motivation for Block~C in this study's architecture.

*Stream I gap:* The M&A failure rate is predictable from structural
information that is systematically absent from existing valuation models.
The following three streams diagnose exactly which information each prior
paradigm discards — and why.

#line(length: 100%, stroke: 0.4pt + gray)

== Stream II — The Tabular Paradigm: An Asymptotic Ceiling

=== The Econometric Era: Linearity Bias and Its Persistence

The first generation of quantitative M&A predictors — Palepu @PALEPU19863,
Barnes @prediction_barnes_1990 — employed logistic regression and multiple
discriminant analysis on financial ratios. These models operate under a
"Linearity Bias": the assumption that financial metrics bear monotonic
relationships to synergy potential. Barnes @prediction_barnes_1990
demonstrated empirically that this assumption fails for leverage: moderate
debt signals fiscal discipline; high debt signals distress. This creates
a non-monotonic, context-dependent relationship that linear discriminants
fundamentally cannot represent.

More critically, Palepu's @PALEPU19863 own out-of-sample validation found
that acquisition targets could not be reliably predicted beyond chance — a
result that exposed the information deficit in ratio-based models rather than
any implementation flaw. The pseudo-R² ceiling below 0.10 documented
consistently across econometric studies @Betton_Eckbo_Thorburn_2008 is not
a calibration failure; it is mathematical evidence that financial ratios
do not contain sufficient information to reconstruct synergy outcomes.
This ceiling is the first benchmark any successor model must demonstrably
surpass.

=== The Machine Learning Escalation: Why XGBoost Hits a Wall

The emergence of ensemble methods promised to resolve the non-linearity
problem. Zhang et al. @aidriven_zhang_2024 deployed random forests and
gradient-boosted decision trees on financial ratio vectors, reporting
accuracy gains over logistic baselines. These improvements are real — but
they are _bounded_ and _misleading_. Four structural problems afflict these
approaches, each of which XGBoost cannot overcome regardless of depth or
regularisation configuration:

*Problem 1 — The Independence Ceiling.* Both random forests and XGBoost
operate on rows of a feature matrix where each row corresponds to one
deal in isolation. The model learns per-deal statistical patterns but
has no mechanism to propagate information across related deals — if two
acquirers share a supplier and that supplier is distressed, both deals
share latent risk that tabular models cannot detect. This is not a
data engineering problem; it is an architectural impossibility.

*Problem 2 — Survivorship Bias.* M&A deal samples drawn from completed
transactions systematically undersample value-destructive deals that were
abandoned or never announced, biasing estimated coefficients toward the
characteristics of deals that _proceeded_ rather than deals that
_succeeded_ @Betton_Eckbo_Thorburn_2008. The consequence is that reported
out-of-sample accuracy reflects the characteristics of deal survival, not
value creation — a fundamental misdirection that no ensemble architecture
can correct without restructuring the sampling frame.

*Problem 3 — Look-Ahead Contamination.* Feature engineering pipelines for
M&A ML studies frequently construct ratios from annual filings that post-date
the deal announcement, embedding future information into the training set.
Zhang et al. @aidriven_zhang_2024, for instance, discretise financial
variables into categorical bins without explicit documentation of fiscal year
alignment with announcement dates. This look-ahead contamination inflates
validation accuracy beyond any economically realisable level.

*Problem 4 — Feature Engineering as Crystallised Bias.* Discretising
continuous financial variables into categorical bins before tree construction
(a common pipeline step in @aidriven_zhang_2024) encodes human analyst
heuristics into irreversible feature representations. The model learns from
a discretised proxy for the raw signal — an automated version of the same
heuristic reasoning employed by the human analysts whose systematic failures
motivated building the model in the first place.

The combined effect of these four problems is an asymptotic accuracy ceiling:
once the linear, rank-order, and non-parametric signal in tabular financial
features has been captured, no further improvement is achievable through
more sophisticated tabular models. This ceiling is _not_ a function of
computational power — it is a function of the information content of the
feature space.

=== The Deep Learning False Dawn: MLPs and the Independence Assumption

The introduction of deep MLPs and LSTMs appeared to dissolve the linearity
constraint. Elhoseny, Metawa et al. @Elhoseny_Metawa_Sztano_El-Hasnony_2022
achieved 95.8% accuracy on financial distress prediction using an
Ameliorative Whale Optimization Algorithm-tuned deep neural network —
a result that superficially suggests deep learning has resolved the
prediction problem. It has not. The model processes each firm as a
standalone entity, consuming individual financial metrics without any
mechanism for inter-firm information exchange. Impressive accuracy on
financial distress classification does not transfer to M&A synergy
prediction: the former asks "is this firm failing?" from its own time
series; the latter asks "will combining two firms create more value than
they possess separately?" — a fundamentally relational question that
non-relational architectures cannot answer.

The same architectural constraint applies to even the most sophisticated
MLP implementations. A model with 10 layers and 10,000 parameters, trained
on a vector of 50 financial ratios, still operates in a 50-dimensional
space that is topologically flat: it contains no representation of
whether the acquirer's top customer is about to go bankrupt, whether the
target and acquirer share a fragile common supplier, or whether the deal
creates a bottleneck in the industrial network. These omissions are not
correctable by adding more layers.

*Stream II gap:* Every generation of tabular model — logistic regression,
random forest, XGBoost, MLP, LSTM — inherits the independence assumption
from its predecessor. The asymptotic ceiling is not a function of model
complexity but of feature space topology. Breaking through it requires a
graph operator (Stream~IV). Before reaching that conclusion, Stream~III
examines why NLP approaches, despite initially appearing to transcend the
tabular paradigm, ultimately reproduce the same structural failure.

#line(length: 100%, stroke: 0.4pt + gray)

== Stream III — The Semantic Turn: Powerful Tools, Wrong Targets

=== The Bag-of-Words Era and the Loughran-McDonald Benchmark

The seminal contribution to financial NLP is Loughran and McDonald
@LOUGHRAN_MCDONALD_2011, who demonstrated that general-purpose sentiment
lexicons systematically misclassify financial language: terms like
"liability," "risk," and "obligation" carry negative sentiment in general
English but are neutral legal descriptors in financial filings. Their
domain-specific word lists corrected this systematic mislabelling and
established a critical methodological principle: financial text requires
finance-specific analytical tools. This insight remains foundational and
is directly inherited by this study's use of FinBERT over general BERT.

However, the bag-of-words (BoW) paradigm itself is architecturally
insufficient beyond this correction. BoW models are "compositionally
blind" @LOUGHRAN_MCDONALD_2011: they cannot distinguish "We have
eliminated our exposure to risk" from "We have significant exposure to
risk," because both sentences produce near-identical word frequency
vectors while conveying directly opposite strategic signals. In the
high-stakes context of M&A filings, where a single conditional clause
("subject to regulatory approval") can invert the meaning of a
multi-billion dollar liability disclosure, this ambiguity is not marginal
but catastrophic. Despite this, recent work continues to extend the BoW
approach: Demers, Wang and Wu @Demers_Wang_Wu_2024 construct frequency-based
lexicons for human capital disclosures; Acheampong et al.
@Acheampong_Mousavi_Gozgor_Yeboah_2025 proxy financial constraints using
lexicon-based indices; Garcia, Hu and Rohrer @Garcia_Hu_Rohrer_2020 develop
finance dictionaries through market-based validation — all approaches that
encode compositional blindness into increasingly refined vocabulary lists.

Hoberg and Phillips @hoberg-phillips-2016 represent the most structurally
sophisticated BoW application: their text-based network industries
classification (TNIC) used product-description cosine similarity from 10-K
filings to construct time-varying firm similarity networks, demonstrating
that textual proximity predicts competitive dynamics and deal _likelihood_
with meaningful accuracy. This study directly inherits the semantic proximity
methodology from Hoberg and Phillips @hoberg-phillips-2016 — but redirects
it from deal occurrence to deal outcome, and replaces TF-IDF vectors with
contextual FinBERT embeddings.

=== The Transformer Revolution and the Limits of Standard FinBERT

Devlin et al.'s @2.3.2-DEVLIN2018 BERT architecture rendered static
lexicon approaches architecturally obsolete by introducing contextual
embeddings: dynamic vector representations where each word's encoding is a
function of its surrounding context, enabling the model to resolve the
polysemy that BoW approaches cannot. Araci's @2.3.2-ARACHI2019 FinBERT
specialised this architecture through pre-training on a large corpus of
corporate financial filings, giving it domain-specific representations that
general BERT models lack.

However, even FinBERT — as deployed in existing M&A studies — falls into
what this review terms the "Tabular Trap." Zhao, Li and Zheng
@zhao2020bertbasedsentimentanalysis extracted BERT-based sentiment from M&A
news articles and then _flattened the rich contextual output into a scalar
sentiment score_ before feeding it into a standard XGBoost classifier. This
pipeline compresses a 768-dimensional semantic embedding — which encodes
rich relational information about entities, relationships, and strategic
context — into a single number, discarding the very representational
richness that motivated using a transformer in the first place. The result
is architecturally identical to an advanced BoW approach: the transformer
acts as a sophisticated tokeniser, but its output is forced through the
same bottleneck that limits lexicon-based methods.

Hajek et al. @2.3.3-Hajek_2024 and Han et al. @2.3.3-Han_2023 repeat this
structural error in their M&A studies: both use transformer architectures
to extract representations from financial text, but both predict binary
deal _occurrence_ (acquisition yes/no) rather than deal _outcome_
(value-creating vs. value-destroying). This distinction is not incidental —
it is the fundamental difference between a market microstructure problem
(when will a deal be announced?) and a corporate strategy problem (will this
deal create economic value?). Predicting deal occurrence requires detecting
announcement patterns in news flow; predicting deal outcome requires
modelling the fundamental complementarity between acquirer and target across
multiple information dimensions. The existing NLP literature has perfected
the former and almost entirely ignored the latter.

=== Why Standard FinBERT Cannot Recover the Full Textual Signal

Beyond the "wrong target" problem, the standard FinBERT implementation
used in the M&A literature has three additional structural limitations that
this study directly addresses:

*Limitation 1 — Frozen Generic Embeddings.* Frozen FinBERT weights,
pre-trained on a general corpus of financial filings, cannot adapt their
representations to the specific semantic dimensions relevant to synergy
prediction — particularly the distinction between _strategic alignment_
(similarity in MD&A disclosures indicating compatible corporate strategies)
and _risk concentration_ (similarity in Risk Factor disclosures indicating
overlapping vulnerabilities). This study exploits this distinction explicitly
in H2, treating the two sections as semantically distinct modalities.

*Limitation 2 — Single-Document Architecture.* Studies that apply FinBERT
to a single company's filing (acquirer or target independently) miss the
_cross-document_ signal: the pairwise semantic distance between acquirer
and target disclosures. It is not the absolute strategic sophistication
of either party that predicts synergy — it is their _relative alignment_
across specific sections. This pairwise delta is what this study computes
via cosine distance between paired acquirer-target embeddings.

*Limitation 3 — Section Conflation.* Standard implementations embed
entire 10-K documents or undifferentiated excerpts. This conflates
sections with opposite hypothesised relationships to CAR: MD&A similarity
(strategic fit → positive CAR) and Risk Factor similarity (risk overlap →
negative CAR). A model that conflates these sections is fitting a noisy
mixture of opposing signals — a source of systematic downward bias on any
textual feature's predictive coefficient.

The choice of FinBERT over Longformer @2.3.2-DEVLIN2018 is justified on
the following grounds: (i) domain-specific pre-training vocabulary
alignment with the 10-K MD&A and Risk Factors sections; (ii) the
section-specific extraction protocol used in this study obviates the
long-document limitation of the 512-token context window; and
(iii) Qin and Yang @qin-yang-2019 demonstrated that frozen domain-specific
representations with lightweight downstream heads yield competitive
performance on financial classification tasks with sample sizes (n~<~5,000)
comparable to this study's deal universe.

*Stream III gap:* Transformer-based NLP architectures possess the semantic
resolution to capture strategic fit signals in 10-K filings. They have not
been applied to post-merger synergy outcome classification. Furthermore, even
when correctly directed, textual models operating on isolated nodes miss the
second-order structural reality: that two firms with perfectly aligned
strategies can still fail to create synergy if their network positions make
the combined entity more fragile. This structural incompleteness is what
Stream~IV addresses.

#line(length: 100%, stroke: 0.4pt + gray)

== Stream IV — The Topological Turn: Network Structure as Irreducible Signal

=== The Economic Foundation: Supply Chain Momentum

The theoretical justification for encoding network topology into synergy
prediction models is grounded in empirical financial economics, not
computational novelty. Cohen and Frazzini @frazzini-cohen-2008 established
the foundational empirical result: economic shocks to a supplier do not
immediately price into the customer firm's equity. Due to information friction
and limited investor attention, these shocks propagate across supply chain
links with a measurable time lag, generating a predictable momentum effect
exploitable by investors who monitor inter-firm dependency structures. This
"supply chain momentum" is direct empirical evidence that network topology
encodes information about future firm value that is absent from standalone
financial metrics — information that by definition cannot be captured by any
model operating on single-firm feature vectors.

Ahern and Harford @ahern-harford-2014 extended this reasoning directly to
M&A, demonstrating that the structure of industry-level trade networks
predicts merger wave propagation: acquisitions cluster along supply chain
linkages because buyers seek to internalise relationships that generate the
highest dependency-reduction value. More critically for this study, Ahern
and Harford @ahern-harford-2014 found that supply chain proximity at the
industry level predicts post-merger combined stock returns — providing
direct precedent that network position contains synergy-relevant information
beyond financial fundamentals. This study operationalises this insight at
the firm level using Bloomberg SPLC data, enabling node-level rather than
industry-level topology encoding.

=== Corporate Network Centrality and Value

The broader corporate networks literature reinforces this foundation.
Fee and Thomas @fee-thomas-2004 documented that customer-supplier
relationships contain systematic pricing power information: firms with high
customer concentration earn excess returns upon resolution of the
relationship, confirming that vertical network dependencies have measurable
equity value implications. Larcker, So, and Wang @larcker-so-wang-2013
demonstrated that director network centrality — betweenness centrality in
board interlocks — is a significant predictor of firm future performance
and acquisition premia, establishing the precedent that graph-theoretic
centrality measures carry economic signal in corporate finance contexts.

The theoretical mechanism connecting these findings to H3 (Topological
Arbitrage Hypothesis) is as follows: acquirers with high betweenness
centrality in the supply-chain-competitor graph occupy bridge positions
between otherwise disconnected industrial communities. These "bridge
nodes" face bilateral dependency constraints — their centrality creates
both upside synergy capture opportunities and downside exposure to
disruption propagation. This bilateral constraint is hypothesised to
compress the variance of $|"CAR"|$ outcomes relative to peripheral nodes,
as the diversifying effect of multiple network connections dampens both
extreme positive and extreme negative market reactions.

=== Why XGBoost Cannot Recover the Topological Signal (Even With Graph Features)

A technically sophisticated critic might argue: cannot supply chain
betweenness centrality simply be added as a hand-engineered feature to an
XGBoost model? This approach — treating centrality as a scalar attribute —
has two fundamental limitations that motivate a full GNN architecture.

First, scalar centrality measures collapse a high-dimensional structural
signal into a single number, discarding the specific _pattern_ of
connectivity that carries economic meaning. Two firms can share identical
betweenness centrality yet have radically different network contexts: one
may bridge two healthy, high-growth industrial clusters; the other may
bridge two sectors in secular decline. A scalar encoding treats these as
identical; a graph embedding that propagates neighbourhood information
does not. Hamilton, Ying, and Leskovec's @3.1HamiltonYL17 GraphSAGE
demonstrated exactly this: by learning aggregation functions over sampled
neighbourhoods, the model recovers structural patterns that any scalar
compression of the same graph would miss.

Second, scalar features cannot propagate _second-order_ network effects.
If an acquirer's primary supplier is itself a customer of the target, this
creates a dependency loop that dramatically affects the risk profile of the
combined entity — but this loop is invisible to any feature engineering
pipeline that processes each pair independently. A message-passing GNN
propagates information through multiple hop neighbourhoods by design,
recovering these higher-order structural patterns without requiring their
explicit enumeration.

Venuti @2.4.2-venutti2021 demonstrated GraphSAGE predicting acquisitions
for private enterprise companies with 81.79% accuracy — concrete evidence
of the inductive architecture's practical applicability to data-scarce
M&A targets. Critically, Venuti's @2.4.2-venutti2021 accuracy gain over
non-graph baselines was _largest_ for private targets with sparse financial
data — precisely the regime where graph neighbourhood information provides
the largest marginal value over standalone features.

=== Heterogeneous Graphs: Why Edge-Type Semantics Matter

A homogeneous GNN that treats all inter-firm edges as equivalent commits
a category error. A `supplies_to` relationship implies operational
dependency and risk propagation: if the supplier fails, the customer's
production line stops. A `competes_with` relationship implies market
concentration and pricing power: if the two competitors merge, their
combined market share may attract regulatory intervention. Collapsing
these semantically distinct relationships into a single undirected edge
type introduces a mixed signal that obscures what each relationship type
independently encodes — analogous to averaging the coefficients of
completely different explanatory variables.

Shi et al. @shi-han-2019 formalised the theoretical basis for
Heterogeneous Information Networks (HINs) and demonstrated that
type-specific attention mechanisms during neighbourhood aggregation
consistently outperform homogeneous baselines on multi-relational graphs.
Wang et al. @3.1wang2021heterogeneousgraphattentionnetwork extended this
into the Heterogeneous Graph Attention Network (HAN), showing that
meta-path-based aggregation recovers semantic structure that single-type
GCNs discard. Lv et al. @lv-etal-2021 provided rigorous benchmarks
confirming that heterogeneous graph encoders achieve statistically
superior performance across multiple classification tasks on multi-relational
corporate networks.

This study therefore constructs a heterogeneous graph
$cal(G) = (cal(V), cal(E), cal(T)_v, cal(T)_e)$ with semantically distinct
edge encodings for `supplier_of`, `customer_of`, `competitor_of`, and
`acquires` relationships within PyTorch Geometric's `HeteroData` structure.
Type-specific GraphSAGE aggregation functions are applied independently
per edge type before cross-type attention pooling — recovering exactly the
semantic distinctions that scalar feature engineering and homogeneous
GCNs discard.

*Stream IV gap:* GNN architectures are theoretically optimal for encoding
inter-firm dependencies, and supply chain network structure is empirically
demonstrated to contain synergy-relevant information. Yet no published study
has directed a heterogeneous GNN at post-merger synergy outcome classification
using firm-level topology. The Topological Alpha Hypothesis (H1), the
Topological Arbitrage Hypothesis (H3), and the HeteroGraphSAGE architecture
proposed in this study constitute the original contribution at this
intersection.

#line(length: 100%, stroke: 0.4pt + gray)

== The Measurement Foundation: Event Study Methodology

=== The Market Model and Its Implementation

The standard empirical tool for measuring M&A value creation is the event
study, formalised by MacKinlay @Mackinlay1997EventSI and whose statistical
properties were rigorously characterised by Brown and Warner
@brown-warner-1985. The methodology rests on the market model: for
each acquirer $i$, a normal returns generating process
$R_(i,t) = alpha_i + beta_i R_(m,t) + epsilon_(i,t)$ is estimated over a
pre-event estimation window, and Abnormal Returns (ARs) are computed as
deviations from this expected return during the event window. The Cumulative
Abnormal Return $"CAR"_i = sum_(t=tau_1)^(tau_2) "AR"_(i,t)$ aggregates
these deviations across the event window $[tau_1, tau_2]$.

MacKinlay @Mackinlay1997EventSI established that event window selection
involves a fundamental bias-variance trade-off: narrow windows
(e.g., $[-1, +1]$) minimise noise from confounding events but risk missing
delayed market reactions for deals with information leakage; wider windows
(e.g., $[-5, +5]$) capture more complete price discovery but increase
variance. This study adopts $[-5, +5]$ trading days, consistent with the
empirical M&A literature @Betton_Eckbo_Thorburn_2008, as a balance that
accommodates pre-announcement leakage (documented to begin 3–5 days before
announcement in approximately 25% of transactions) while limiting
contamination from post-announcement deal renegotiations.

=== Critical Assumptions and Their Limitations

Brown and Warner @brown-warner-1985 identified three critical assumptions
underlying the market model that warrant explicit scrutiny in M&A contexts.
First, the _stationarity assumption_ — that $beta_i$ is stable between the
estimation window and the event window — is potentially violated for
acquirers strategically repositioning ahead of the deal announcement.
This study enforces an estimation window of $[-250, -10]$ with a 120-day
minimum observation threshold, following MacKinlay's @Mackinlay1997EventSI
recommendation, with deals below this threshold excluded from the final
sample.

Second, the _benchmark contamination_ risk: for mega-cap acquirers whose
deal is large enough to shift the S&P~500 index, using the same index as
the expected-returns benchmark creates a circularity. This study uses the
value-weighted CRSP market index rather than the S&P~500 specifically to
reduce single-stock concentration effects on the benchmark.

Third, Fama's @fama-1991 review of efficient markets evidence is
particularly germane: the semi-strong form efficiency assumption embedded
in event studies implies that announcement-date CARs represent unbiased
estimates of the market's assessment of deal value. However, Fama
@fama-1991 also documented systematic M&A anomalies — including long-run
post-merger underperformance — inconsistent with strong-form efficiency.
This study treats CAR as a noisy but informative signal of synergy
potential, not a perfect measure, and uses binary directional
classification rather than continuous regression to reduce sensitivity
to measurement noise @Betton_Eckbo_Thorburn_2008.

=== Justification for Binary Classification

The signal-to-noise characteristics of short-window CARs make continuous
regression a challenging target for machine learning models. The
pseudo-R² ceiling below 0.10 documented consistently across econometric
studies @Betton_Eckbo_Thorburn_2008 @PALEPU19863 is consistent with the
Efficient Market Hypothesis's implication @fama-1991 that the _magnitude_
of market surprise is largely unpredictable from pre-announcement
information. The _direction_ of CAR, however, contains learnable signal
rooted in deal fundamentals — whether financial capacity is aligned with
strategic intent and ecosystem health — that the feature space can
realistically support. This study therefore defines $y_i = 1$ if
$"CAR"_i > 0$ (value-creating) and $y_i = 0$ otherwise (value-destructive),
with AUC-ROC as the primary metric for its robustness to class imbalance
and threshold-invariant evaluation of discriminative power @ajayi_2022.

#line(length: 100%, stroke: 0.4pt + gray)

== The Multimodal Imperative: Why Fusion Is the Only Exit

=== Complementary Variance and the Failure of Mono-Modal Models

The four streams surveyed above converge on a single structural finding:
financial, textual, and topological features encode _different_, partially
non-overlapping aspects of synergy potential — what Baltrušaitis, Ahuja,
and Morency @3.1Baltrušaitis-Ahuja formalise as "complementary variance"
in their taxonomy of multimodal learning architectures. A firm may exhibit
strong capital adequacy (financial signal), precise strategic alignment
(textual signal), yet suffer from fragile supplier dependencies
(topological signal). A model observing only one dimension is
mathematically blind to the divergence between the others, and no
within-modality engineering can recover information that was never
measured.

Xu et al. @xu-etal-2021 demonstrated the practical validity of this
argument in a financial forecasting context, showing that structured
financial data and unstructured text features carry statistically
independent predictive signal — signal that vanishes from each modality
when the other is controlled for, but is recoverable when both are jointly
modelled. Qin and Yang @qin-yang-2019 similarly demonstrated complementarity
between price-based and news-based features for stock movement prediction.
Taken together, these results provide empirical support for the multimodal
fusion architecture adopted in this study.

=== Why HeteroGraphSAGE Fusion Is the Only Logical Exit

Standard baselines — logistic regression (Stream~II), XGBoost on ratios
(Stream~II), standalone FinBERT classifier (Stream~III), homogeneous GCN
with scalar centrality features (Stream~IV) — each fail for a specific,
architecturally irreversible reason. The table in @tab:synthesis maps
each failure mode precisely.

The HeteroGraphSAGE fusion architecture proposed in this study is designed
to resolve each failure mode simultaneously: (i) GraphSAGE's inductive
neighbourhood aggregation recovers the topological signal that tabular
models cannot access; (ii) FinBERT's contextual embeddings on section-split
10-K filings recover the semantic signal that BoW and scalar NLP cannot
represent; (iii) late fusion via concatenation preserves modality-specific
representations while enabling cross-modal learning in the joint prediction
head; and (iv) heterogeneous edge-type encoding preserves the semantic
distinctions between `supplier_of`, `customer_of`, and `competitor_of`
relationships that homogeneous GCNs discard.

This is not an incremental improvement over existing methods — it is a
structural departure from the shared independence assumption that has
constrained every prior generation of M&A predictive model. The
following ablation design (Chapters~3 and~4) is specifically constructed
to test whether this structural departure yields the statistically
measurable accuracy gains that the theoretical argument predicts.

=== Architectural Choice: Late Fusion and Its Justification

The M&A deal universe with complete multimodal coverage is approximately
2,800–5,000 observations — far below the sample requirements for stable
joint end-to-end training of transformer and GNN components
@3.1Baltrušaitis-Ahuja. This study therefore adopts _late fusion_: each
modality is encoded independently into a fixed-dimensional embedding vector
($bold(h)_F in RR^d_F$, $bold(h)_T in RR^d_T$, $bold(h)_G in RR^d_G$),
and these vectors are concatenated into a joint representation
$bold(z)_i = [bold(h)_F || bold(h)_T || bold(h)_G]$ before a shared
MLP prediction head. This design isolates modality-specific representation
learning from cross-modal inference, enabling robust individual stream
training even when certain modalities have incomplete coverage — a practical
necessity given that SPLC network data and EDGAR filings are not available
for all deals in the LSEG sample.

The prediction head uses Chen and Guestrin's @3.1Chen_2016 XGBoost
framework for the baseline ablation experiments, given its documented
superiority over neural networks on heterogeneous tabular feature vectors
at M&A sample sizes. The full multimodal MLP prediction head incorporates
dropout regularisation @3.1JMLR:v15:srivastava14a to control overfitting
given the high feature-to-sample ratio. SHAP decomposition
@3.1ying2019gnnexplainergeneratingexplanationsgraph is applied
post-inference to provide per-modality attribution scores, enabling the
controlled ablation experiments to quantify the marginal contribution of
each block to predictive accuracy.

#line(length: 100%, stroke: 0.4pt + gray)

== Synthesis: Gap Table and Research Hypotheses

The four knowledge streams converge on the synthesis table below, which
maps each prior study to its structural failure mode and the specific
architectural component of this study designed to overcome it.

#figure(
  caption: [
    Synthesis of Four Literature Streams: Methods, Structural Failure
    Modes, and Architectural Responses
  ],
  table(
    columns: (1.7fr, 1.4fr, 2.1fr, 1.8fr),
    align: (left, left, left, left),
    inset: 6pt,
    stroke: 0.4pt,
    table.header(
      [*Study / Stream*], [*Method*], [*Structural Failure Mode*],
      [*This Study's Response*]
    ),
    [Palepu (1986); Barnes (1990)],
    [Logit / MDA on ratios],
    [Linearity bias; pseudo-R² < 0.10; no network features],
    [Blocks A+B+C; non-linear fusion; graph topology],

    [Zhang et al. (2024)],
    [Random forest, XGBoost],
    [i.i.d. assumption; survivorship bias; look-ahead contamination],
    [Temporal train/val/test splits; graphSAGE neighbourhood aggregation],

    [Elhoseny et al. (2022)],
    [Deep MLP (AWOA-DL)],
    [High accuracy on distress; independence assumption unchanged; wrong target],
    [Relational graph operator; synergy CAR as target],

    [Zhao et al. (2020); Han et al. (2023)],
    [BERT / RoBERTa → XGBoost],
    [Tabular trap: embeddings flattened; predicts deal occurrence, not outcome],
    [FinBERT on section-split 10-Ks; binary CAR classification (H2)],

    [Hajek et al. (2024)],
    [FinBERT sentiment],
    [Single-document; no pairwise delta; acquisition likelihood target],
    [Pairwise acquirer-target cosine distance; H2 conditional directionality],

    [Loughran \& McDonald (2011); Hoberg \& Phillips (2016)],
    [BoW / TF-IDF similarity],
    [Compositional blindness; deal occurrence target],
    [Contextual FinBERT embeddings; section-specific similarity (H2)],

    [Cohen \& Frazzini (2008); Ahern \& Harford (2014)],
    [Industry-level supply chain analysis],
    [Industry-level only; no firm-level GNN; no CAR prediction],
    [Firm-level GraphSAGE on SPLC data (H1, H3)],

    [Venuti (2021)],
    [Homogeneous GraphSAGE],
    [Predicts deal likelihood; homogeneous edge types; no text or financial fusion],
    [Heterogeneous edge types; full multimodal fusion],

    [Alochukwu et al. (2024)],
    [Dynamic GNN],
    [Volatility prediction target; no M&A synergy outcome],
    [CAR direction as prediction target; HeteroGraphSAGE architecture],

    [Baltrušaitis et al. (2019); Xu et al. (2021)],
    [Multimodal fusion frameworks],
    [Not applied to M&A synergy; no graph modality],
    [Full Block A+B+C late fusion with SHAP decomposition],
  )
) <tab:synthesis>

No prior published study has jointly fused financial fundamentals,
FinBERT textual embeddings, and supply-chain graph topology, directed
this multimodal architecture at binary CAR direction classification as
a measure of post-merger synergy, and tested the marginal contribution
of each modality through controlled ablation. This trifecta of novelty
constitutes the primary contribution of this dissertation.

=== Formal Research Hypotheses

The following three hypotheses emerge directly from the four-stream
synthesis. Each is operationalised to map precisely to an ablation
experiment in the empirical design.

*H1 — The Topological Alpha Hypothesis:*
The inclusion of second-order neighbour embeddings via GraphSAGE
(Block~C) will yield a statistically significant increase in AUC-ROC
($p < 0.05$, paired Diebold-Mariano test) relative to the
financial-only baseline (Block~A), under 5-fold stratified
cross-validation. This gain will be disproportionately concentrated
within supply-chain-dependent manufacturing sectors (SIC~20–49)
compared to asset-light service sectors (SIC~60–79), reflecting the
hypothesis that graph embeddings recover signal proportional to
industrial ecosystem structural density @frazzini-cohen-2008
@ahern-harford-2014.

*H2 — The Semantic Divergence Hypothesis:*
The predictive relationship between textual similarity and synergy
direction is _conditional_ on document section. High cosine similarity
in strategic disclosures (MD&A) will positively correlate with
$"CAR"$ — capturing strategic fit — whereas high similarity in Risk
Factors will negatively correlate with $"CAR"$ — capturing risk
concentration. This conditional directionality is tested via bivariate
OLS $"CAR"_i = beta_0 + beta_1 dot "sim"_("MDA",i) + beta_2 dot
"sim"_("RF",i) + epsilon_i$, and directly refutes the monotonic sentiment
utility assumption embedded in @LOUGHRAN_MCDONALD_2011 and
@2.3.3-Hajek_2024.

*H3 — The Topological Arbitrage Hypothesis:*
Acquirer nodes exhibiting high betweenness centrality in the
heterogeneous supply-chain graph will exhibit statistically compressed
variance in $|"CAR"|$ outcomes relative to peripheral nodes, as
measured by Levene's test across betweenness centrality quantile
groups @larcker-so-wang-2013 @ahern-harford-2014. The full multimodal
fusion model (Blocks A+B+C) will achieve statistically superior
AUC-ROC relative to all single-modality baselines, confirming the
multimodal complementarity established by @3.1Baltrušaitis-Ahuja
and @xu-etal-2021.
