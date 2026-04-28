
// ============================================================
//  02-lit-review.typ
//  M&A Synergy Prediction — Literature Review
//  Hard Joshi | Student ID: 2512658
//  University of East London
// ============================================================

= Literature Review

== The Central Argument: A Gap Three Decades Wide

Existing quantitative frameworks for predicting post-acquisition synergy are fundamentally
constrained by a shared architectural assumption: they treat the firm as an isolated data point.
Financial ratio models, machine learning classifiers trained on tabular data, and even
transformer-based NLP pipelines all process firms as nodes without edges — independent and
identically distributed objects divorced from the industrial ecosystems that define their actual
value. This literature review argues, through a systematic critique of four interconnected
knowledge streams, that this independence assumption is not merely a modelling convenience
but the root cause of the persistent failure of M&A valuation. The review builds toward a
single conclusion: accurate synergy prediction requires the simultaneous fusion of financial
fundamentals (Block~A), textual strategic intent (Block~B), and network-structural topology
(Block~C) within a heterogeneous graph framework — a configuration that, as of 2025, no
published study has deployed for post-merger synergy outcome prediction.

Each section below is therefore not a neutral survey but a motivated critique. Prior work is
assessed not only for what it achieves, but for what it structurally _cannot_ achieve, and
how each limitation directly motivates one of the three hypotheses tested in this study.

#line(length: 100%, stroke: 0.4pt + gray)

== The M&A Paradox: Systematic Failure as a Starting Point

=== The Empirical Record of Value Destruction

Mergers and Acquisitions remain the primary mechanism for global capital reallocation, yet
their empirical track record constitutes one of the most robust anomalies in corporate
finance. Martynova and Renneboog @MARTYNOVA20082148 documented that between 70% and 90%
of acquisitions fail to generate value for acquirer shareholders across a full century of
global takeover activity, a finding replicated in multiple independent meta-analyses
@Christensen_Alton_Rising_Waldeck_2011. Critically, this rate of failure has exhibited
statistical stationarity across distinct economic regimes — from the conglomerate wave of
the 1960s through the dot-com era and post-financial-crisis consolidations — implying that
the root cause is structural rather than cyclical @MARTYNOVA20082148.

Bradley, Desai, and Kim @BRADLEY19883 established that while target shareholders captured
average announcement-date gains of approximately 30%, acquirer shareholders experienced
corresponding losses, yielding combined synergy gains averaging only 7.4% — a figure
representing the total economic surplus that sophisticated predictive models should, in
principle, be able to identify _ex ante_. The persistence of this destruction in the face
of increasingly sophisticated due diligence practices motivates the epistemological
question this study addresses: what information, structurally absent from conventional
models, would close this predictive gap?

=== Behavioural Distortions: Hubris as Label Noise

The first layer of explanation is behavioural. Roll's Hubris Hypothesis @Roll1986TheHH
argues that acquiring managers systematically overestimate their ability to extract value,
treating the target's efficient market price as an underestimate rather than an unbiased
signal. This creates the "Synergy Trap" identified by Sirower @sirower_synergy_1997,
wherein the premium paid mathematically necessitates performance improvements that are,
in the majority of cases, unattainable.

From a machine learning perspective, hubris introduces systematic _label noise_ into
training datasets. If acquisition premiums partly reflect executive overconfidence rather
than genuine synergy potential @Roll1986TheHH, then models trained on transaction values
are fitting a signal contaminated with behavioural error. This motivates the decision to
use Cumulative Abnormal Return (CAR) — a market-derived measure of investor surprise
net of expected returns — as the prediction target rather than deal premia or accounting
synergies, since CAR represents the market's _ex post_ revision of expectations and is
less susceptible to managerial narrative.

=== Market Timing and the Stock-Driven Acquisition Wave

A complementary source of label contamination is identified by Shleifer and Vishny
@shleifer-vishny-2003, whose stock-market-driven acquisitions model demonstrates that
overvalued acquirers rationally exploit inflated equity as acquisition currency during
market peaks. Under this framework, a substantial proportion of M&A waves are motivated
by relative misvaluation rather than operational complementarity — the observed CAR
reflects market correction dynamics rather than synergy realisation. This finding
reinforces the argument that isolating _direction_ of CAR (positive vs. negative) is a
more tractable and economically meaningful prediction target than its precise magnitude,
since directional classification is less sensitive to the market-timing noise that
confounds continuous regression @Betton_Eckbo_Thorburn_2008.

=== The Epistemological Barrier: Information Asymmetry and Structural Blindness

Even setting aside behavioural distortions, the verification mechanism itself is
structurally deficient. Angwin @angwin_mergers_2001 documented that conventional
due diligence operates in organisational silos — financial, legal, and operational
teams independently assess their respective domains without modelling interaction effects.
This orthogonal approach systematically misses multiplicative risk: a target may exhibit
strong cash flows _and_ a fragile single-source supplier dependency; in linear summation
the deal passes scrutiny, but the combined exposure renders the cash flow non-existent.

Akerlof's @akerlof-1970 "Market for Lemons" framework provides the theoretical
grounding for this structural blindness: the seller possesses superior information
about asset quality, and mandated disclosures (10-K filings) are insufficient to
eliminate this asymmetry since targets can strategically obscure ecosystem vulnerabilities
within the noise of standard reporting formats @Hansen1987. The implication is
direct: reliable synergy estimation requires information sources that cannot be
strategically manipulated — specifically, the verifiable topology of a firm's external
network of suppliers, customers, and competitors. This observation forms the primary
theoretical motivation for Block~C (graph features) in this study's architecture.

*Gap identified:* The M&A failure rate is not random; it is predictable from structural
information absent from all existing predictive frameworks. The remainder of this review
diagnoses precisely which information each prior paradigm discards.

#line(length: 100%, stroke: 0.4pt + gray)

== The Tabular Paradigm: Computational Evolution Without Architectural Progress

=== The Econometric Era and the Fallacy of Linearity

The first generation of quantitative M&A predictors, established by Palepu @PALEPU19863,
employed logistic regression to predict acquisition likelihood from financial ratios.
These models operated under what we term a "Linearity Bias" — the assumption that
financial metrics bear monotonic relationships to synergy potential. Barnes @prediction_barnes_1990
demonstrated that this assumption fails empirically: moderate leverage signals financial
discipline, while high leverage signals distress, producing a non-linear relationship
that linear discriminant models fundamentally misinterpret.

More critically, Palepu's @PALEPU19863 own validation revealed that acquisition targets
could not be reliably predicted, with out-of-sample accuracy barely exceeding chance —
a result that exposed the fundamental information deficit in ratio-based models rather
than any implementation flaw. The pseudo-R² ceiling below 0.10 documented across
multiple econometric studies @Betton_Eckbo_Thorburn_2008 is not a calibration failure;
it is evidence that financial ratios do not contain sufficient information to reconstruct
synergy outcomes.

=== The Machine Learning Escalation: Sophistication Without Structural Change

The emergence of ensemble methods promised to resolve the non-linearity problem. Recent
studies such as Zhang et al. @aidriven_zhang_2024 deployed random forests and gradient
boosting on financial ratio vectors, reporting accuracy gains over logistic baselines.
However, a critical examination reveals a persistent architectural stagnation: the
computational complexity of the models increased, while the fundamental independence
assumption — that each firm is an i.i.d. data point — remained unchanged. More
problematically, Zhang et al. @aidriven_zhang_2024 discretised continuous financial
variables into categorical bins during feature engineering, crystallising human analyst
heuristics into irreversible feature representations and effectively automating the
same linearity bias their models purport to transcend.

Luypaert and De Maeseneire's predictive models for acquisition likelihood similarly
suffered from survivorship bias: deal samples drawn from completed transactions
systematically undersample value-destructive deals that were abandoned or never
announced, biasing estimated coefficients toward the characteristics of deals that
_proceeded_ rather than deals that _succeeded_ @Betton_Eckbo_Thorburn_2008. This
look-ahead contamination means that out-of-sample performance is routinely overstated
in the M&A ML literature — a methodological flaw this study addresses through strict
temporal train/validation/test splits that respect the chronological order of
announcement dates.

=== The Persistent Tabular Trap

Even NLP-augmented approaches fall into what we term the "Tabular Trap." Zhao,
Li, and Zheng @zhao2020bertbasedsentimentanalysis employed BERT to extract sentiment
from M&A news but flattened the rich contextual output into a scalar vector for
downstream XGBoost classification, discarding the relational structure of mentioned
entities. The model treated textual features as attributes of isolated nodes —
the fundamental independence assumption in a new guise. Similarly, Elhoseny,
Metawa et al. @Elhoseny_Metawa_Sztano_El-Hasnony_2022 achieved 95.8% accuracy on
financial distress prediction using deep neural networks, yet the architecture
processed each firm as a standalone entity, explicitly ignoring contagion dynamics
from distressed network partners.

*Gap identified:* The tabular paradigm — whether implemented via logistic regression,
random forests, or MLP — cannot recover the signal encoded in inter-firm dependencies.
No matter how sophisticated the within-node feature engineering, the structural
information loss is mathematically irreversible without a graph operator. This
motivates the inclusion of Block~C and directly grounds H1 (Topological Alpha Hypothesis).

#line(length: 100%, stroke: 0.4pt + gray)

== The Measurement of Market Reaction: Event Study Methodology

=== Foundations and the Market Model

The standard empirical tool for measuring M&A value creation is the event study,
formalised by MacKinlay @Mackinlay1997EventSI and whose statistical properties were
rigorously characterised by Brown and Warner @brown-warner-1985. The methodology rests
on the market model: for each acquirer $i$, a normal returns generating process
$R_(i,t) = alpha_i + beta_i R_(m,t) + epsilon_(i,t)$ is estimated over a pre-event
estimation window, and Abnormal Returns (ARs) are computed as deviations from this
expected return during the event window. The Cumulative Abnormal Return
$"CAR"_i = sum_(t=tau_1)^(tau_2) "AR"_(i,t)$ aggregates these deviations across
the event window $[tau_1, tau_2]$.

MacKinlay @Mackinlay1997EventSI established that the choice of event window involves
a fundamental bias-variance trade-off. Narrow windows (e.g., $[-1, +1]$) minimise
noise from confounding events but risk missing delayed market reactions for deals
with information leakage. Wider windows (e.g., $[-5, +5]$) capture more complete
price discovery but introduce greater variance. This study adopts $[-5, +5]$ trading
days, consistent with the empirical M&A literature @Betton_Eckbo_Thorburn_2008, as
a balance that accommodates pre-announcement leakage (documented to begin 3-5 days
before announcement in approximately 25% of transactions) while limiting contamination
from post-announcement deal renegotiations.

=== Critical Assumptions and Their Limitations

Brown and Warner @brown-warner-1985 identified three critical assumptions underlying
the market model that warrant scrutiny in M&A contexts. First, the stationarity
assumption — that $beta_i$ is stable between the estimation window and the event window
— is potentially violated for acquirers who have been strategically repositioning ahead
of the deal announcement, as operational changes alter the firm's systematic risk
exposure. Second, the choice of benchmark (this study uses the S&P~500 index for US
acquirers) introduces benchmark contamination risk when the deal itself is large enough
to affect the index composition. Third, the OLS estimation requires a minimum number
of trading days to produce reliable coefficient estimates; this study enforces a 120-day
minimum in the estimation window $[-250, -10]$, following MacKinlay's @Mackinlay1997EventSI
recommended threshold, with deals below this threshold excluded from the sample.

Fama's @fama-1991 review of efficient markets evidence is particularly germane here:
the semi-strong form efficiency assumption embedded in event studies implies that
announcement-date CARs represent unbiased estimates of the market's assessment of
deal value. However, Fama @fama-1991 also documented systematic M&A anomalies —
including long-run post-merger underperformance — that are inconsistent with strong-form
efficiency. This study therefore treats CAR as a noisy but informative signal of
synergy potential, not as a perfect measure, and uses binary directional classification
rather than continuous regression to reduce sensitivity to measurement noise
@Betton_Eckbo_Thorburn_2008.

=== Justification for Binary Classification Target

The signal-to-noise characteristics of short-window CARs make continuous regression
a challenging target for machine learning models. Pseudo-R² values below 0.10 are
consistently documented across econometric studies of M&A predictability
@Betton_Eckbo_Thorburn_2008 @PALEPU19863, consistent with the Efficient Market
Hypothesis's implication that the _magnitude_ of market surprise is largely
unpredictable from pre-announcement information. The _direction_ of CAR, however,
contains learnable signal rooted in deal fundamentals — whether financial capacity
is aligned with strategic intent and ecosystem health — that the feature space
can realistically support.

This study therefore reframes the prediction target as binary classification:
$y_i = 1$ if $"CAR"_i > 0$ (value-creating) and $y_i = 0$ otherwise
(value-destructive). AUC-ROC is adopted as the primary metric over Accuracy
and F1 due to its robustness to class imbalance and its threshold-invariant
evaluation of discriminative power @ajayi_2022.

*Gap identified:* Prior event study methodology provides a rigorous framework for
measuring market reaction but has not been integrated with multimodal predictive
models. The gap is not in the measurement instrument but in its connection to a
feature-rich prediction architecture — the bridge this study constructs.

#line(length: 100%, stroke: 0.4pt + gray)

== Textual Analysis in Finance: The Semantic Turn and Its Incomplete Revolution

=== The Bag-of-Words Era and Compositional Blindness

The seminal contribution to financial NLP is Loughran and McDonald @LOUGHRAN_MCDONALD_2011,
who demonstrated that general-purpose sentiment lexicons systematically misclassify
financial language — terms like "liability" and "risk" that carry negative connotations
in general English are neutral legal descriptors in financial filings. Their construction
of domain-specific word lists represented a significant methodological advance and
established the principle that financial text requires finance-specific analytical tools.

However, Bag-of-Words (BoW) approaches remain "compositionally blind" @LOUGHRAN_MCDONALD_2011.
Word-frequency models cannot distinguish "We have eliminated exposure to risk" from "We
have significant exposure to risk" — the word counts are nearly identical while the
meaning is inverted. In the high-stakes context of M&A, where a single conditional
clause can invert the meaning of a multi-billion dollar liability disclosure, this
ambiguity is not a marginal limitation but a fundamental failure mode. Despite this,
recent work continues to extend BoW approaches: Demers, Wang and Wu @Demers_Wang_Wu_2024
constructed frequency-based lexicons for human capital disclosures, while Acheampong
et al. @Acheampong_Mousavi_Gozgor_Yeboah_2025 proxied financial constraints using
lexicon-based indices — approaches that encode the same compositional blindness
into more domain-specific vocabulary.

Hoberg and Phillips @hoberg-phillips-2016 represent a more structurally sophisticated
BoW application: their text-based network industries classification (TNIC) used
product-description similarity from 10-K filings to construct time-varying firm
similarity networks, demonstrating that textual proximity is a meaningful predictor
of competitive dynamics and deal likelihood. Critically, however, Hoberg and Phillips
@hoberg-phillips-2016 predicted deal _occurrence_ rather than deal _outcome_ — an
important distinction addressed in Section 2.4 below.

=== The Transformer Revolution and FinBERT

The release of BERT by Devlin et al. @2.3.2-DEVLIN2018 rendered static lexicon
approaches architecturally obsolete by introducing contextual embeddings — dynamic
vector representations where a word's encoding is a function of its surrounding
context, enabling the model to resolve polysemy that BoW approaches cannot.
Araci's @2.3.2-ARACHI2019 FinBERT specialised this architecture through pre-training
on a large corpus of corporate financial filings, giving it domain-specific
representations of financial language that general BERT models lack.

The choice of FinBERT over alternative architectures — including RoBERTa, Longformer,
or general BERT-base — requires explicit justification given the limitations of frozen
transformer embeddings for long documents. Three considerations motivate this selection.
First, FinBERT's domain-specific pre-training corpus aligns with the 10-K MD&A sections
used in this study, reducing the risk of domain mismatch that would affect a general
language model applied to financial text @2.3.2-ARACHI2019. Second, for the semantic
divergence computation in H2, the _relative_ distances between embeddings are more
important than absolute representation quality; FinBERT's consistent financial vocabulary
encoding makes cosine similarity between acquirer and target embeddings a meaningful
measure of strategic alignment. Third, FinBERT's 512-token context window, while
insufficient for full-document processing, is adequate for the MD&A and Risk Factors
sections when extracted independently, which this study does via section-specific
EDGAR parsing.

The limitation of frozen embeddings — that the projection head cannot adapt the
representations to the synergy prediction task — is acknowledged explicitly: the
FinBERT weights are frozen to prevent overfitting on a sample of approximately
2,800–5,000 deals, far below the data requirements for stable fine-tuning of a
110M-parameter model. Qin and Yang @qin-yang-2019 demonstrated that frozen
pre-trained representations, when combined with lightweight downstream heads, yield
competitive performance on financial classification tasks with small training sets —
a finding that supports this architectural choice.

=== The "Wrong Target" Problem in NLP-Based M&A Research

Despite the availability of powerful semantic tools, their application to M&A synergy
prediction remains fragmented. Hajek et al. @2.3.3-Hajek_2024 employed FinBERT for
M&A news sentiment prediction, achieving strong results for binary "acquisition event"
prediction. Han et al. @2.3.3-Han_2023 used RoBERTa to identify acquisition targets
from textual disclosures. Both studies demonstrate that NLP tools contain predictive
signal — but both are directed at _deal occurrence_ rather than _deal outcome_.

This distinction is critical. Predicting whether a deal will happen is a market
microstructure problem; predicting whether it will create value is a corporate strategy
problem. The former can be addressed by detecting announcement patterns in news flow;
the latter requires modelling the fundamental complementarity between acquirer and target
— their strategic fit, risk alignment, and ecosystem compatibility. There is a conspicuous
absence of research applying domain-specific transformer embeddings to the classification
of post-merger synergy direction (positive vs. negative CAR). This absence constitutes
the direct motivation for H2 (Semantic Divergence Hypothesis).

*Gap identified:* Transformer architectures possess the semantic resolution to capture
strategic fit signals in 10-K filings. They have not been applied to post-merger
synergy classification. Furthermore, no prior study has tested the _conditional_ nature
of the textual signal: whether strategic-section similarity (MD&A) and risk-section
similarity (Risk Factors) have opposite directional relationships with CAR — the
specific claim of H2.

#line(length: 100%, stroke: 0.4pt + gray)

== The Topological Turn: Network Structure as Economic Signal

=== The Economic Basis: Supply Chain Momentum and Information Friction

The theoretical justification for encoding network topology into synergy prediction
models is grounded in empirical financial economics, not computational novelty.
Cohen and Frazzini @frazzini-cohen-2008 established the foundational result: economic
shocks to a supplier do not immediately price into the customer firm's equity due to
information friction and limited investor attention. Instead, these shocks propagate
across supply chain links with a measurable time lag, generating a predictable momentum
effect exploitable by investors who monitor inter-firm dependency structures. This
"supply chain momentum" provides direct empirical evidence that network topology
encodes information about future firm value that is not reflected in standalone
financial metrics.

Ahern and Harford @ahern-harford-2014 extended this reasoning specifically to M&A,
demonstrating that the structure of industry-level trade networks predicts merger wave
propagation: acquisitions cluster along supply chain linkages because buyers seek to
internalise supply chain relationships that generate the highest dependency-reduction
value. Critically, Ahern and Harford @ahern-harford-2014 found that supply chain
proximity at the industry level predicts post-merger combined stock returns — direct
evidence that network position contains synergy-relevant information beyond financial
fundamentals. This study operationalises this insight at the firm level using
Bloomberg SPLC data, enabling node-level rather than industry-level topology encoding.

=== Corporate Network Centrality and Value

The broader corporate networks literature provides additional grounding. Fee and Thomas
@fee-thomas-2004 documented that customer-supplier relationships contain systematic
pricing power information: firms with high customer concentration earn excess returns
upon resolution of the relationship, confirming that vertical network dependencies have
measurable equity value implications. Larcker, So, and Wang @larcker-so-wang-2013
demonstrated that director network centrality — betweenness centrality in board
interlocks — is a significant predictor of firm future performance and acquisition
premia, establishing the precedent that graph-theoretic centrality measures carry
economic signal in corporate finance contexts.

The theoretical mechanism connecting these findings to this study's H3 (Topological
Arbitrage Hypothesis) runs as follows: acquirers with high betweenness centrality
in the supply-chain-competitor graph occupy bridge positions between otherwise
disconnected industrial communities. These firms face bilateral dependency constraints
— their network position creates both upside synergy capture opportunities (access to
multiple supply chains) and downside exposure to disruption propagation. This bilateral
constraint is hypothesised to compress the variance of |CAR| outcomes relative to
peripheral nodes, as the diversifying effect of multiple network connections dampens
both extreme positive and extreme negative market reactions.

=== Graph Neural Networks: From Transductive to Inductive Learning

To operationalise network topology as a machine-learnable feature, this study employs
Graph Neural Networks (GNNs). The foundational architecture established by Kipf and
Welling @kipf-welling-2017 — Graph Convolutional Networks (GCNs) — demonstrated that
node representations can be enriched through recursive neighbourhood aggregation,
achieving state-of-the-art performance on citation and social networks. However,
GCNs are _transductive_: they require all nodes to be present during training,
rendering them incapable of generating embeddings for firms absent from the training
graph — a critical limitation in M&A contexts where targets may be private companies
or newly listed entities.

Hamilton, Ying, and Leskovec @3.1HamiltonYL17 resolved this through GraphSAGE
(Sample and AggreGatE), which learns aggregation _functions_ over sampled neighbourhoods
rather than memorising fixed node embeddings. This inductive capability allows
GraphSAGE to generate representations for completely unseen nodes by aggregating
information from their network context, which is directly applicable to private M&A
targets whose neighbourhood (customers, suppliers, competitors of their industry peers)
is often observable even when the target itself lacks historical data. Venuti
@2.4.2-venutti2021 demonstrated GraphSAGE predicting acquisitions for private enterprise
companies with 81.79% accuracy — concrete evidence of the inductive architecture's
practical applicability to data-scarce M&A targets.

=== Heterogeneous Information Networks: The Semantic Gap in Graph Structure

A homogeneous GNN that treats all inter-firm edges as equivalent commits a category
error: a `supplies_to` relationship implies operational dependency and risk propagation,
while a `competes_with` relationship implies market concentration and pricing power.
Collapsing these semantically distinct edge types into a single edge type introduces
noise that obscures the signal each relationship type separately encodes.

Shi et al. @shi-han-2019 formalised the theoretical basis for Heterogeneous Information
Networks (HINs) and introduced the Heterogeneous Graph Attention Network (HAN), which
applies type-specific attention mechanisms to node aggregation across different
relationship types. Wang et al. @3.1wang2021heterogeneousgraphattentionnetwork
extended this into a practical Heterogeneous Graph Attention Network architecture
demonstrating that type-specific aggregation consistently outperforms homogeneous
baselines on network classification tasks. Lv et al. @lv-etal-2021 further demonstrated
that heterogeneous graph encoders capture richer semantic structures in knowledge graphs,
providing the theoretical basis for their application to multi-relational corporate
networks.

This study therefore constructs a heterogeneous graph $cal(G) = (cal(V), cal(E), cal(T)_v, cal(T)_e)$
using PyTorch Geometric's `HeteroData` structure, with distinct edge type encodings
for `supplier_of`, `customer_of`, `competitor_of`, and `acquires` relationships. This
preserves the semantic information that homogeneous GCN implementations would discard.

=== The "Wrong Target" Problem in GNN-Based Finance Research

Despite substantial technical progress, GNN applications in finance exhibit the
same misdirection observed in NLP research: sophisticated topology is pointed at
the wrong prediction target. Alochukwu @2.4.3-wang2024 deployed dynamic GNNs to
predict stock market volatility with high temporal fidelity; Wasi et al.
@2.4.3-he2023graphneuralnetworkssupply demonstrated GNNs predicting supply chain
contagion risk. Both studies target market _activity_ (price change, contagion
probability) rather than corporate _productivity_ (post-merger value creation).

There is no published study that links supply chain and competitor network topology
to post-merger synergy classification via CAR direction. The field has perfected
the prediction of "who will acquire whom" and "how will the market move" but has
not asked the strategically critical question: "will this specific deal create value,
given the network positions of both parties?" This gap is the primary research
contribution of this study.

*Gap identified:* GNN architectures are theoretically optimal for encoding inter-firm
dependencies. They have not been applied to post-merger synergy outcome classification.
Furthermore, the specific hypothesis that network centrality predicts CAR variance
compression (H3) has no direct precedent in the literature — constituting the novel
empirical contribution of this study.

#line(length: 100%, stroke: 0.4pt + gray)

== Multimodal Fusion: The Imperative for Integration

=== Complementary Variance and the Failure of Mono-Modal Models

The literature surveyed above establishes that financial, textual, and topological
features encode different, partially non-overlapping aspects of synergy potential.
This property — which Baltrušaitis, Ahuja, and Morency @3.1Baltrušaitis-Ahuja
formalise as "complementary variance" in their taxonomy of multimodal learning
architectures — implies that any mono-modal model incurs an irreducible information
loss. A firm may exhibit strong capital adequacy (financial signal) yet suffer from
strategic incoherence (textual signal) and fragile supplier dependencies (topological
signal); a model observing only one dimension is mathematically blind to the divergence.

Xu et al. @xu-etal-2021 demonstrated the practical validity of this theoretical
argument in a financial forecasting context, showing that structured financial data
and unstructured text features carry statistically independent predictive signal
for earnings announcement returns — signal that vanishes from each modality when
the other is available, but is recoverable when both are jointly modelled. Qin and
Yang @qin-yang-2019 similarly demonstrated complementarity between price-based
and news-based features for stock movement prediction. Taken together, these results
provide empirical support for the multimodal architecture adopted in this study.

=== Architectural Choice: Late Fusion and Its Justification

The architectural question is not whether to fuse but _how_ to fuse. End-to-end
joint training theoretically permits cross-modal gradient flow, allowing the model
to learn cross-modal interactions. However, this advantage is negated when sample
sizes fall below the stability threshold for deep learning. The M&A deal universe
with complete multimodal coverage is approximately 2,800–5,000 observations
— far below the typical requirements for stable joint training of transformer and
GNN components @3.1Baltrušaitis-Ahuja.

This study therefore adopts late fusion: each modality is encoded independently
into a fixed-dimensional embedding vector, and these vectors are concatenated
into a joint representation $z_i = [h_F || h_T || h_G]$ before a shared prediction
head. This design isolates modality-specific representation learning from cross-modal
inference, enabling robust individual stream training even when certain modalities
have incomplete coverage — a practical necessity given that SPLC network data
and EDGAR filings are not available for all deals in the LSEG sample.

Chen and Guestrin's @3.1Chen_2016 XGBoost framework is used as the prediction
head for the baseline ablation experiments, given its documented superiority
over neural networks on heterogeneous tabular vectors with the sample sizes
typical in M&A research. The fusion MLP prediction head in the primary model
incorporates dropout regularisation @3.1JMLR:v15:srivastava14a to control
overfitting given the high feature-to-sample ratio.

=== Interpretability: SHAP Decomposition of Modality Contributions

Financial applications of machine learning require interpretability beyond
aggregate performance metrics. Lundberg and Lee @lundberg-lee-2017 introduced
SHAP (SHapley Additive exPlanations), grounded in cooperative game theory, which
decomposes individual model predictions into per-feature contributions with
desirable theoretical properties (efficiency, symmetry, dummy) that alternative
importance measures such as permutation importance and attention weights do not
satisfy. Recent work has demonstrated that attention weights in transformer
architectures are unreliable proxies for feature importance @3.1ying2019gnnexplainergeneratingexplanationsgraph,
strengthening the case for post-hoc SHAP decomposition as the interpretability
mechanism for this study's multimodal architecture.

#line(length: 100%, stroke: 0.4pt + gray)

== Synthesis and Research Gap Table

The preceding sections establish a landscape of "methodological orthogonality":
each prior paradigm captures one dimension of synergy potential while discarding
the others. The table below maps prior work to its structural limitations and
the specific research gap each leaves open.

#figure(
  caption: [Synthesis of Prior Literature: Methods, Limitations, and Research Gaps],
  table(
    columns: (1.8fr, 1.5fr, 2fr, 1.7fr),
    align: (left, left, left, left),
    inset: 6pt,
    stroke: 0.4pt,
    table.header(
      [*Study / Stream*], [*Method*], [*Limitation*], [*Gap Addressed Here*]
    ),
    [Palepu (1986); Barnes (1990)], [Logit/MDA on ratios], [Linearity bias; pseudo-R² < 0.10], [Non-linear fusion with graph topology],
    [Zhang et al. (2024)], [Random forest, XGBoost], [i.i.d. assumption; no network features; look-ahead bias risk], [Temporal splits; Block C topology],
    [Zhao et al. (2020); Han et al. (2023)], [BERT/RoBERTa classifiers], [Tabular trap; predicts deal occurrence, not outcome], [FinBERT on MD&A for synergy direction (H2)],
    [Loughran & McDonald (2011); Hoberg & Phillips (2016)], [BoW / TF-IDF similarity], [Compositional blindness; no deal outcome prediction], [Contextual FinBERT embeddings],
    [Cohen & Frazzini (2008); Ahern & Harford (2014)], [Industry-level supply chain analysis], [Industry-level only; no firm-level GNN encoding], [Firm-level GraphSAGE on SPLC data (H1, H3)],
    [Venuti (2021); Alochukwu (2024)], [GraphSAGE; dynamic GNNs], [Predicts acquisition likelihood / volatility, not synergy CAR], [GraphSAGE → CAR direction classification],
    [Baltrušaitis et al. (2019); Xu et al. (2021)], [Late fusion frameworks], [Not applied to M&A synergy outcome], [Full multimodal fusion (H3)],
  )
) <tab:synthesis>

No prior published study has (i) jointly fused financial fundamentals, FinBERT
textual embeddings, and supply-chain graph topology, (ii) directed this multimodal
architecture at binary CAR direction classification as a measure of post-merger
synergy, and (iii) tested the conditional information value of each modality through
controlled ablation. This trifecta of novelty constitutes the primary contribution
of this dissertation.

#line(length: 100%, stroke: 0.4pt + gray)

== Research Hypotheses

The following three hypotheses emerge directly from the gaps identified in the
synthesis above. Each is precisely operationalised to map to a specific ablation
experiment in the empirical design.

*H1 — The Topological Alpha Hypothesis:*
The inclusion of second-order neighbour embeddings via GraphSAGE (Block~C) will
yield a statistically significant increase in AUC-ROC ($p < 0.05$, paired
Diebold-Mariano test) for binary synergy classification relative to the
financial-only baseline (Block~A), under 5-fold stratified cross-validation.
This gain will be disproportionately concentrated within supply-chain-dependent
manufacturing sectors (SIC 20–49) compared to asset-light service sectors (SIC 60–79),
reflecting the hypothesis that graph embeddings recover signal proportional to the
structural density of the industrial ecosystem @frazzini-cohen-2008 @ahern-harford-2014.

*H2 — The Semantic Divergence Hypothesis:*
The predictive relationship between textual similarity and synergy direction is
_conditional_ on document section. High cosine similarity in strategic disclosures
(MD&A) will positively correlate with CAR — capturing strategic fit — whereas
high similarity in Risk Factors will negatively correlate with CAR — capturing
risk concentration. This conditional directionality, tested via bivariate OLS
($"CAR"_i = beta_0 + beta_1 dot "sim"_("MDA",i) + beta_2 dot "sim"_("RF",i) + epsilon_i$),
refutes the monotonic sentiment utility assumption of prior NLP classifiers
@LOUGHRAN_MCDONALD_2011 @2.3.3-Hajek_2024.

*H3 — The Topological Arbitrage Hypothesis:*
Acquirer nodes exhibiting high betweenness centrality in the heterogeneous
supply-chain graph will exhibit statistically compressed variance in $|"CAR"|$
outcomes relative to peripheral nodes, as measured by Levene's test across
betweenness centrality quantile groups. This compression reflects the bilateral
dependency constraints of bridge nodes, which dampen both upside and downside
market reactions @larcker-so-wang-2013 @ahern-harford-2014. The full multimodal
fusion model (Blocks A+B+C) will achieve statistically superior AUC-ROC relative
to all single-modality baselines, confirming the multimodal imperative established
by Baltrušaitis et al. @3.1Baltrušaitis-Ahuja and Xu et al. @xu-etal-2021.
