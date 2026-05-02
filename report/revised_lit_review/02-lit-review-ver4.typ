// ============================================================
//  02-lit-review-ver4.typ  — Final Submission Draft
//  M&A Synergy Prediction — Literature Review
//  Hard Joshi | Student ID: 2512658 | University of East London
// ============================================================

= Literature Review

== Introduction

Every year, corporations spend hundreds of billions of dollars acquiring other companies — and most of those acquisitions destroy value rather than create it. The failure rate has remained stubbornly between 70% and 90% for decades @martynova2008, a persistence that cannot be explained by bad luck or difficult market conditions. Something more fundamental is wrong with how acquisitions are evaluated.

This literature review diagnoses that failure. It traces a critical arc through four generations of quantitative M&A research — from early financial ratio models through machine learning classifiers, then through natural language processing applied to company filings, and finally through graph-based network models. At each stage, we will identify what the approach gains and, crucially, what it structurally loses. The review builds toward a single argument: that synergy is a latent variable recoverable only from the intersection of three complementary information channels — financial fundamentals, textual disclosures, and supply chain topology — and that no single-modality approach has yet captured this intersection.

The review is structured as four interconnected streams. Stream I establishes the M&A paradox: why value destruction is so persistent and what information gaps make prediction so difficult. Stream II examines the tabular paradigm: how financial ratios and machine learning models trained on them hit a definable ceiling because they treat every firm as an isolated data point. Stream III examines the semantic turn: how textual analysis of filings adds strategic context but systematically targets the wrong prediction problem. Stream IV examines the topological turn: how supply chain networks encode irreducible information about firm health that no standalone feature can capture. The review concludes with a synthesis section that maps every structural gap in prior work to a specific architectural response in this study, and formally states three testable hypotheses.

#line(length: 100%, stroke: 0.4pt + gray)

== Stream I — The M&A Paradox: Why Prediction Is Hard

The fundamental empirical observation motivating this study is straightforward: most acquisitions destroy value for acquiring shareholders, and this has been true for as long as researchers have been measuring it.

=== The Empirical Record

Martynova and Renneboog @martynova2008 documented that between 70% and 90% of acquisitions failed to generate positive returns for acquirer shareholders across a full century of global takeover activity. This finding has been replicated in multiple independent studies and meta-analyses @christensen2011. The striking feature of this failure rate is not its magnitude — high failure rates in complex corporate transactions are expectable — but its *stationarity*: the rate has not meaningfully improved despite decades of increasingly sophisticated analytical tools. This implies the problem is not cyclical or situational but rooted in the structure of how acquisitions are evaluated.

Bradley, Desai, and Kim @bradley1988 decomposed who captures the gains from takeovers and found a crucial asymmetry: target shareholders earn approximately 30% average abnormal returns around announcement, while acquirer shareholders systematically lose. The combined entity creates value — synergy exists — but the gains flow almost entirely to targets. An acquirer attempting to capture synergy by paying a premium faces a mathematical challenge: the premium paid must be recovered from the present value of future synergies, and that present value is very difficult to estimate reliably in advance. Sirower @sirower1997 formalised this as the "Synergy Trap": the premium required often exceeds what any realistic synergy realisation can repay.

=== Hubris as Systematic Label Noise

The first and most intuitive explanation for persistent failure is managerial hubris. Roll @roll1986 proposed that acquiring managers systematically overestimate their ability to create value from acquisitions, mistaking the target's market price for an undervaluation signal rather than an efficient market consensus. The behavioural bias is not merely an anecdote — it introduces a specific technical problem for quantitative modelling. If acquisition premiums partly reflect executive overconfidence rather than genuine synergy potential, then models trained to predict synergy from deal characteristics are fitting a contaminated target variable. The "label" — whether a deal created or destroyed value — is noisy in a specific direction: toward overoptimism.

This matters methodologically. Label noise does not merely increase variance; it introduces bias that no amount of model sophistication can eliminate, because the bias is encoded in the target variable itself. Any model trained to predict deal value from transaction features is partially predicting managerial ego.

=== Market Timing and the Stock-Driven Wave

A second, complementary source of contamination is identified by Shleifer and Vishny @shleifer2003. Their "stock market driven acquisitions" model demonstrates that firms with overvalued equity have a rational incentive to use that equity as acquisition currency during market peaks — not because the target represents genuine operational complementarity, but because the acquirer's shares are expensive and can purchase more implied value than a rigorous fundamental analysis would justify. Under this framework, a substantial fraction of M&A waves are driven by relative misvaluation rather than synergy potential, and the resulting announcement returns reflect market corrections rather than genuine synergy realisation.

This is not merely a behavioural explanation — it has a structural implication. If significant portions of M&A activity are market-timing exercises rather than value-creation exercises, then pre-announcement fundamental features cannot reliably predict announcement returns, because the returns are partly determined by how overvalued the acquirer's stock was at the time of the deal rather than by the intrinsic quality of the target.

=== The Epistemological Barrier: Structural Blindness

Even setting aside behavioural and market-timing noise, the verification process itself is structurally deficient. Angwin @angwin2001 documented that conventional due diligence operates in organisational silos: financial teams analyse balance sheets, legal teams scrutinise contracts, and operational teams assess supply chains independently, without modelling the interactions between these domains.

This orthogonal approach misses multiplicative risk. Consider a target with strong cash flows but a fragile single-source supplier dependency. Linear due diligence passes both assessments independently — the cash flows look fine, the supplier risk is flagged as a note. But in combination, the two assessments reveal near-zero value: collapse the supplier, eliminate the cash flows. No team modelled the conjunction.

Akerlof's @akerlof1970 "Market for Lemons" framework explains why this structural blindness persists: the target possesses superior information about its own vulnerabilities, and mandated disclosures are insufficient to reveal all material weaknesses. Targets can strategically obscure structural risks within the volume of standard reporting. This creates an epistemological barrier: the information needed to accurately value a target may be precisely the information the target has the greatest incentive to conceal.

This observation generates the central methodological implication of Stream I: *reliable synergy estimation requires information that cannot be strategically manipulated — specifically, the verifiable structure of the firm's external network of suppliers, customers, and competitors.* This observation directly motivates the inclusion of supply chain topology (Stream IV) in this study's multimodal framework, and it establishes why financial features alone cannot resolve the prediction problem.

*Stream I summary:* The M&A failure rate is not random — it is driven by behavioural hubris (label noise), market timing (return contamination), and information asymmetry (structural blindness). Each of these failure modes requires a specific information channel to address: behavioural noise requires market-derived labels rather than deal premium; market timing contamination requires models robust to macro-level confounding; structural blindness requires external verifiable network data rather than self-reported filings. These requirements jointly justify the multimodal approach adopted in this study.

#line(length: 100%, stroke: 0.4pt + gray)

== Stream II — The Tabular Paradigm: An Asymptotic Ceiling

The failure of subjective valuation created a vacuum that quantitative models attempted to fill. The history of M&A prediction is a story of increasing computational sophistication under a shared structural assumption: that each firm can be treated as an independent, isolated data point. This section argues that this assumption — the "independence assumption" — is not merely a modelling convenience but the fundamental reason that tabular approaches have hit a definable accuracy ceiling.

=== The Econometric Era: Why Linear Models Cannot Represent Financial Relationships

The first generation of quantitative M&A models, pioneered by Palepu @palepu1986, employed logistic regression and multiple discriminant analysis on financial ratios to predict acquisition likelihood. These models operated under an implicit "linearity bias": the assumption that financial metrics bear monotonic relationships to synergy potential. Higher growth is better; lower leverage is safer; larger market share implies more pricing power.

Barnes @barnes1990 demonstrated that this assumption fails empirically. Moderate leverage signals financial discipline; high leverage signals distress. The relationship between a financial ratio and synergy potential is non-monotonic, context-dependent, and often non-linear. Linear discriminant models fundamentally cannot represent such relationships.

More critically, Palepu's @palepu1986 own out-of-sample validation found that acquisition targets could not be reliably predicted beyond chance levels. The pseudo-R² ceiling consistently documented across econometric studies @betton2008 — values almost never exceeding 0.10 — is not evidence that the models need better calibration. It is mathematical evidence that financial ratios alone do not contain sufficient information to distinguish synergy-creating deals from value-destructive ones. This ceiling is the first benchmark any successor model must demonstrably surpass to make a meaningful contribution.

=== Why Machine Learning Does Not Escape the Independence Assumption

The emergence of ensemble methods — random forests and gradient-boosted decision trees — appeared to dissolve the linearity constraint. These models can represent arbitrary non-linear interactions between features, which should in principle allow them to capture the non-monotonic relationships that defeated linear discriminant analysis.

Zhang et al. @zhang2024 deployed random forests and XGBoost on financial ratio vectors, reporting improved accuracy over logistic baselines. These improvements are real. However, they are bounded, and understanding why requires examining the architecture rather than the algorithm.

Both random forests and gradient-boosted trees operate on a feature matrix where each row corresponds to one deal in isolation. The model learns statistical patterns within individual deal vectors but has no mechanism to propagate information across related deals. If two acquirers share a supplier, and that supplier is financially distressed, both deals face latent risk that no amount of within-deal feature engineering can detect. This is not a data engineering problem that more variables or better preprocessing would fix — it is an architectural impossibility.

Four additional structural problems compound the independence ceiling:

*Survivorship bias* afflicts deal samples drawn from completed transactions. Deals that were abandoned, withdrawn, or never announced are systematically excluded, biasing estimated coefficients toward the characteristics of deals that *proceeded* rather than deals that *succeeded* @betton2008. A model trained on completed deals learns the profile of deal survival, not deal value creation — a fundamental misdirection.

*Look-ahead contamination* occurs when feature engineering pipelines construct ratios from annual financial filings that post-date the deal announcement. The model is inadvertently trained on future information, and any reported out-of-sample accuracy is inflated beyond economically realisable levels.

*Feature discretisation as crystallised bias* — a common preprocessing step that converts continuous financial ratios into categorical bins — encodes human analyst heuristics into irreversible feature representations, automating the same flawed reasoning that motivated building quantitative models in the first place.

The combined effect is an asymptotic accuracy ceiling: once the non-linear, rank-order, and tabular signal in financial features has been captured, no further improvement is achievable through more sophisticated tabular models. This ceiling is not a function of computational power or algorithm choice — it is a function of the information content of the feature space.

=== MLPs and Deep Learning: A Different Architecture, the Same Assumption

The introduction of deep neural networks appeared to offer a structural departure from the tabular paradigm. Elhoseny et al. @elhoseny2022 achieved high accuracy on financial distress prediction using a deep neural network optimised with a whale optimisation algorithm.

However, this accuracy does not transfer to M&A synergy prediction. Financial distress prediction is a time-series classification problem: given a firm's historical financials, predict whether it will fail. M&A synergy prediction is a relational problem: given two firms and their inter-relationship, predict whether their combination creates more value than they possess separately. Even a 10-layer MLP consuming 50 financial ratios per firm still operates in a 50-dimensional space that is topologically flat — it contains no representation of whether the target's supplier is about to fail, whether the two firms share a common customer, or whether the deal creates a structural bottleneck in their industry network. These omissions are not correctable by adding more layers or parameters.

*Stream II summary:* Every generation of tabular model — logistic regression, MDA, random forest, XGBoost, MLP — inherits the independence assumption from its predecessor. The asymptotic ceiling is not a function of model complexity. Breaking through it requires a fundamentally different data structure: a graph.

#line(length: 100%, stroke: 0.4pt + gray)

== Stream III — The Semantic Turn: Powerful Tools, Wrong Targets

The first two streams established that financial features alone cannot solve the M&A prediction problem. The question then becomes: what other information channels are available, and how can they be exploited?

The most natural answer is textual: companies are required to file annual 10-K reports with the Securities and Exchange Commission (SEC), which contain vast quantities of qualitative information about strategy, risk, and operations that no financial ratio can capture. This section traces the evolution of financial text analysis from simple word-counting to sophisticated transformer models, and identifies why even the most advanced textual approaches have not yet been applied to the right problem.

=== The Bag-of-Words Era and Compositional Blindness

The foundational contribution to financial NLP is Loughran and McDonald @loughran2011, who demonstrated that generic sentiment lexicons systematically misclassify financial language. The word "liability" carries negative sentiment in general English but is a neutral legal descriptor in corporate filings. Terms like "risk," "obligation," and "uncertainty" are standard disclosure language, not negative signals. Their domain-specific word lists corrected this systematic mislabelling and established the critical methodological principle: financial text requires finance-specific analytical tools.

However, bag-of-words (BoW) approaches face a more fundamental problem: they are "compositionally blind." A BoW model sees a document as a collection of individual word frequencies, discarding syntax, grammar, and context. It cannot distinguish "We have eliminated our exposure to risk" from "We have significant exposure to risk" — both sentences produce near-identical word frequency vectors while conveying opposite strategic signals. In the high-stakes context of multi-billion dollar acquisitions, where a single conditional clause can invert the meaning of a liability disclosure, this blindness is not marginal but potentially catastrophic.

Despite this limitation, recent work continues to extend BoW approaches to corporate disclosures. Demers et al. @demers2024 construct frequency-based lexicons for human capital disclosures; Acheampong et al. @acheampong2025 proxy financial constraints using lexicon-based indices; Garcia et al. @garcia2020 develop finance dictionaries through market-based validation. All encode compositional blindness into increasingly refined vocabulary lists, refinements that address the Loughran-McDonald correction but not the fundamental architectural limitation.

Hoberg and Phillips @hoberg2016 represent the most structurally sophisticated BoW application: their text-based network industries classification used product-description cosine similarity from 10-K filings to construct time-varying firm similarity networks, demonstrating that textual proximity predicts competitive dynamics and acquisition likelihood. This study inherits the semantic proximity methodology from Hoberg and Phillips but redirects it from deal occurrence to deal outcome, and replaces TF-IDF vectors with contextual embeddings from a domain-specific transformer.

=== The Transformer Revolution and Why Even FinBERT Falls into the Tabular Trap

Devlin et al. @devlin2018 introduced BERT, which rendered static lexicon approaches architecturally obsolete through contextual embeddings — dynamic vector representations where each word's encoding is a function of its surrounding text, enabling polysemy resolution that BoW approaches cannot achieve. Araci @araci2019 specialised BERT for the financial domain by pre-training on corporate filings, producing FinBERT with domain-specific representations of financial language.

However, even FinBERT, as deployed in existing M&A research, falls into what this review terms the "Tabular Trap." Zhao et al. @zhao2020 extracted BERT-based sentiment from M&A news and then *flattened the rich contextual output into a single scalar sentiment score* before feeding it into an XGBoost classifier. This compression discards the 768-dimensional semantic embedding — which encodes rich relational information about entities, relationships, and strategic context — in favour of a single number. The result is architecturally identical to an advanced BoW approach: the transformer acts as a sophisticated tokeniser, but its output is forced through the same representational bottleneck.

More critically, existing M&A studies using transformer NLP consistently target the wrong prediction problem. Hajek et al. @hajek2024 and Han et al. @han2023 both use transformer architectures to predict binary acquisition *likelihood* — whether a deal will happen — rather than acquisition *outcome* — whether the deal will create value. The distinction is fundamental. Predicting deal occurrence is a market microstructure problem: it requires detecting announcement patterns, media coverage, and insider positioning. Predicting deal outcome requires modelling the fundamental complementarity between acquirer and target — their strategic fit, risk alignment, and ecosystem compatibility. The field has largely perfected the former and largely ignored the latter.

=== Why Standard FinBERT Cannot Recover the Full Textual Signal

Beyond the wrong target problem, the standard FinBERT implementation in existing M&A literature has three additional structural limitations that this study directly addresses.

First, *frozen generic embeddings*: FinBERT weights pre-trained on general financial filings cannot adapt their representations to the specific semantic dimensions relevant to synergy prediction, particularly the distinction between strategic alignment in MD&A disclosures (similar strategic narratives suggesting compatible management approaches) and risk concentration in Risk Factor disclosures (similar vulnerability profiles suggesting overlapping exposure to common threats). These two sections encode fundamentally different economic signals, yet standard implementations treat the filing as a single document.

Second, *single-document architecture*: Studies applying FinBERT to a single company's filing miss the cross-document signal. Synergy is not determined by the absolute strategic sophistication of either party — it is determined by their *relative alignment* across specific disclosure sections. Two firms can have very similar MD&A language indicating good strategic fit; they can also have very similar Risk Factor language indicating dangerous risk concentration. Computing the pairwise distance between acquirer and target embeddings across specific sections is what H2 (the Semantic Divergence Hypothesis) operationalises.

Third, *section conflation*: Standard implementations embed entire 10-K documents or undifferentiated excerpts, mixing sections with opposite hypothesised relationships to CAR. A model that conflates strategic alignment (which should positively predict synergy) with risk concentration (which should negatively predict synergy) is fitting a noisy mixture of opposing signals. Section-specific extraction and separate embedding is therefore not merely an enhancement but a methodological necessity.

*Stream III summary:* Transformer architectures possess the semantic resolution to capture strategic fit signals in 10-K filings. They have not been applied to post-merger synergy outcome classification, and even when correctly targeted, they require section-specific processing to avoid signal cancellation. Crucially, text-based models operating on isolated firm nodes miss the second-order structural reality: that two firms with perfectly aligned strategies can still fail to create synergy if their combined network position is more fragile than either firm's individual position. This structural incompleteness motivates Stream IV.

#line(length: 100%, stroke: 0.4pt + gray)

== Stream IV — The Topological Turn: Network Structure as Irreducible Signal

The first three streams established that financial ratios cannot capture synergy, that NLP alone targets the wrong problem, and that both approaches treat firms as isolated entities. This stream argues that firm value is fundamentally relational — defined by a company's position within an ecosystem of suppliers, customers, and competitors — and that this relational information is mathematically unrecoverable from any model that treats firms as independent data points.

=== The Economic Foundation: Supply Chain Momentum

The theoretical justification for encoding network topology is grounded in empirical financial economics, not computational novelty. Cohen and Frazzini @cohen2008 established the foundational result: economic shocks to a supplier do not immediately price into the customer firm's equity due to information friction and limited investor attention. These shocks propagate across supply chain links with a measurable time lag, generating a predictable "supply chain momentum" effect exploitable by investors who monitor inter-firm dependency structures. This is direct empirical evidence that network topology encodes information about future firm value that standalone financial metrics cannot capture.

Ahern and Harford @ahern2014 extended this reasoning directly to M&A, demonstrating that the structure of industry-level trade networks predicts merger wave propagation — acquisitions cluster along supply chain linkages because buyers seek to internalise relationships that generate the highest dependency-reduction value. Critically, they found that supply chain proximity at the industry level predicts post-merger combined stock returns, providing direct precedent that network position contains synergy-relevant information beyond financial fundamentals. This study operationalises this insight at the firm level using Bloomberg Supply Chain data, enabling node-level rather than industry-level topology encoding.

Fee and Thomas @fee2004 documented that customer-supplier relationships contain systematic pricing power information, and Larcker et al. @larcker2013 demonstrated that graph-theoretic centrality measures in corporate networks carry economic signal. Together, these establish that the specific structural position a firm occupies in its industry network — not just its standalone financial metrics — is economically meaningful and quantifiable.

=== Why Only a Graph Operator Can Recover the Topological Signal

A technically sophisticated objection might argue: cannot supply chain centrality simply be computed as a scalar and added as a feature to an XGBoost model? This approach has two fundamental limitations.

First, scalar centrality measures collapse a high-dimensional structural signal into a single number, discarding the specific *pattern* of connectivity that carries economic meaning. Two firms can have identical betweenness centrality yet have radically different network contexts: one may bridge two healthy, high-growth industrial clusters; the other may bridge two sectors in secular decline. A scalar encoding treats these as identical; a graph embedding that propagates neighbourhood information does not.

Second, scalar features cannot propagate *second-order* network effects. If an acquirer's primary supplier is itself a customer of the target, this creates a dependency loop that dramatically affects the risk profile of the combined entity — but this loop is invisible to any feature engineering pipeline that processes each pair independently. Only a message-passing graph neural network that recursively propagates information through multi-hop neighbourhoods can recover these higher-order structural patterns.

=== From Transductive to Inductive Learning: Why GraphSAGE

The choice of graph architecture matters critically. Kipf and Welling @kipf2017 introduced Graph Convolutional Networks (GCNs), which demonstrated that node representations could be enriched through recursive neighbourhood aggregation — achieving state-of-the-art results on citation and social networks. However, GCNs are *transductive*: they require all nodes to be present during training and cannot generate embeddings for firms absent from the training graph.

In M&A contexts, the cold-start problem is pervasive: private companies, newly listed entities, and rarely-traded targets often have sparse or absent historical data. Hamilton et al. @hamilton2017 resolved this through GraphSAGE, which learns *aggregation functions* rather than memorising specific node embeddings. This inductive capability allows GraphSAGE to generate representations for completely unseen nodes by aggregating information from their observable network neighbourhood — directly applicable to private M&A targets whose surrounding network (suppliers, customers, competitors of peer firms) is often observable even when the target itself lacks historical data. Venuti @venuti2021 demonstrated GraphSAGE predicting acquisitions for private enterprise companies with 81.79% accuracy, proving practical applicability to data-scarce M&A targets.

=== Heterogeneous Graphs: Why Edge Type Semantics Matter

A homogeneous graph treats all inter-firm edges as equivalent. This is a category error. A `supplies_to` relationship implies operational dependency and risk propagation: if the supplier fails, the customer's production line stops. A `competes_with` relationship implies market concentration and pricing power: if two competitors merge, their combined market share may attract regulatory scrutiny. Collapsing these semantically distinct relationship types into a single undirected edge introduces mixed signal analogous to averaging coefficients from completely different regressions.

This study therefore constructs a *heterogeneous* graph with semantically distinct edge encodings for `supplier_of`, `customer_of`, `competitor_of`, and `acquires` relationships, applying type-specific GraphSAGE aggregation functions independently per edge type before cross-type attention pooling. This architecture preserves exactly the semantic distinctions that scalar feature engineering and homogeneous graph models discard.

*Stream IV summary:* Supply chain and competitor network topology encodes irreducible information about firm value and synergy potential. Graph neural networks are the only model class capable of recovering this signal. GraphSAGE is selected over alternatives because it can generalise to unseen firms (inductive learning), handle the heterogeneous edge semantics this study requires, and scale to large industrial networks. No published study has applied this architecture to post-merger synergy outcome classification.

#line(length: 100%, stroke: 0.4pt + gray)

== The Multimodal Imperative: Why Fusion Is the Only Exit

The four streams surveyed above converge on a single structural finding: financial, textual, and topological features encode *different*, partially non-overlapping aspects of synergy potential. This property — what Baltrušaitis et al. @baltrusaitis2019 formalise as "complementary variance" in multimodal learning — means that any mono-modal model incurs an irreducible information loss.

A firm may simultaneously exhibit strong capital adequacy (financial signal), precise strategic alignment (textual signal), and fragile supplier dependencies (topological signal). A model observing only the financial dimension is mathematically blind to the other two; a model observing financial and text but not topology misses the structural risk entirely. This is not an engineering gap — it is a theoretical impossibility that no amount of within-modality sophistication can overcome.

Xu et al. @xu2021 demonstrated the practical validity of this argument in financial forecasting, showing that structured financial data and unstructured text features carry statistically independent predictive signal — signal that vanishes from each modality when the other is controlled for, but is recoverable when both are jointly modelled. This provides empirical support for the multimodal architecture this study adopts.

=== Architectural Choice: Late Fusion and Its Justification

A critical design question is whether to train the system end-to-end — jointly optimising text, graph, and classification components on the CAR target — or to pre-compute each modality's representations independently and fuse them later.

End-to-end joint training is theoretically attractive: it allows cross-modal gradient flow, enabling each component to adapt its representations to the downstream task. However, this advantage requires large sample sizes to realise — typically in the hundreds of thousands or millions — because each joint parameter update provides only a tiny contribution to the overall loss. The M&A deal universe with complete multimodal coverage is approximately 2,800 to 5,000 observations — far below the threshold for stable end-to-end training of transformer and GNN components simultaneously.

This study therefore adopts *late fusion*: each modality is encoded independently into a fixed-dimensional embedding vector (financial: 56 dimensions; textual: 128 dimensions; topological: 64 dimensions), and these vectors are concatenated into a joint representation $bold(z)_i = [bold(h)_F || bold(h)_T || bold(h)_G]$ before a shared XGBoost prediction head. This design isolates modality-specific representation learning from cross-modal inference and enables robust individual stream training even when certain modalities have incomplete coverage — a practical necessity given that supply chain data and SEC filings are not available for all deals.

XGBoost is selected as the prediction head over alternative ensemble methods (Random Forest, LightGBM) for three reasons: built-in L1/L2 regularisation provides structural overfitting protection critical when the feature-to-sample ratio is high (249 features to ~2,800 training observations); native handling of missing values accommodates the incomplete modality coverage across the deal sample; and documented superiority over neural networks on heterogeneous tabular feature vectors at this sample size @chen2016.

SHAP decomposition @lundberg2017 provides post-hoc per-modality attribution scores, enabling the controlled ablation experiments to quantify the marginal contribution of each block to predictive accuracy. SHAP is preferred over permutation importance because it is grounded in cooperative game theory (Shapley values), providing desirable theoretical properties — efficiency, symmetry, and dummy — that alternative importance measures do not simultaneously satisfy.

*Stream V summary:* The Multimodal Imperative is not a design preference but a theoretical requirement. Synergy is latent within the intersection of three complementary modalities. No single modality contains sufficient information; no within-modality engineering can recover information that was never measured. The ablation design tests this claim directly by comparing performance across progressively richer modality combinations.

#line(length: 100%, stroke: 0.4pt + gray)

== Event Study Methodology: Measuring What We Are Trying to Predict

Before presenting the research hypotheses, we must establish how the target variable — post-merger synergy — is measured. This section is not a digression; it is foundational, because the measurement method determines what the model is actually being asked to predict.

=== The Market Model and Cumulative Abnormal Return

The standard empirical tool for measuring M&A value creation is the event study, formalised by MacKinlay @mackinlay1997. The methodology uses a *market model* to estimate what an acquirer's returns should have been, had the deal not occurred, and then measures the deviation — the *abnormal return* — during the announcement period.

For each acquirer $i$, a normal returns generating process is estimated over a pre-event estimation window of approximately 250 trading days:

$ R_(i,t) = alpha_i + beta_i R_(m,t) + epsilon_(i,t) $

where $R_(i,t)$ is the acquirer's return on day $t$, $R_(m,t)$ is the market return, and $alpha_i$, $beta_i$ are OLS-estimated parameters. Abnormal returns during the event window are then:

$ A R_(i,t) = R_(i,t) - (hat(alpha)_i + hat(beta)_i R_(m,t)) $

and the Cumulative Abnormal Return (CAR) over the event window is:

$ C A R_i = sum_(t=tau_1)^(tau_2) A R_(i,t) $

This study uses a symmetric 11-day event window $[-5, +5]$ trading days around the announcement date, chosen to accommodate pre-announcement information leakage (documented to begin 3–5 days before announcement in approximately 25% of transactions) while limiting contamination from post-announcement confounding events @mackinlay1997 @betton2008.

The theoretical justification for CAR as a synergy proxy rests on semi-strong market efficiency @fama1991: at the announcement, all publicly available information about the deal is reflected in the stock price, and the market's revaluation represents its collective assessment of whether the combined entity will create or destroy value. The mean CAR of −1.27% across the deal sample is consistent with the empirical finding that acquirers typically experience slight negative announcement returns — reflecting the well-documented "winner's curse" and hubris effects identified in Stream I.

=== Why Binary Classification Rather Than Continuous Regression

A critical methodological choice is the decision to predict the *direction* of CAR (positive versus negative) rather than its *continuous magnitude*. This choice deserves explicit justification.

The pseudo-R² ceiling below 0.10 documented consistently across M&A econometric studies @betton2008 is consistent with the Efficient Market Hypothesis: the precise *magnitude* of market surprise is largely unpredictable from pre-announcement information. Market timing noise, bidder overvaluation, competing bids, and announcement-specific sentiment shocks all contribute variance that has nothing to do with the underlying synergy potential. This makes continuous regression targets intractable for machine learning models.

The *direction* of CAR, however, contains learnable signal rooted in deal fundamentals — whether financial capacity is aligned with strategic intent and ecosystem health — that the feature space can realistically support. This study therefore defines the binary target:

$ y_i = 1 $ if $"CAR"_i > 0$ (value-creating deal)
$ y_i = 0 $ otherwise (value-destructive deal)

with AUC-ROC as the primary evaluation metric, preferred over Accuracy for its robustness to class imbalance (the dataset is moderately imbalanced at 44%/56%) and its threshold-invariant evaluation of discriminative power @betton2008. AUC-ROC measures the probability that a randomly chosen value-creating deal ranks above a randomly chosen value-destructive deal — exactly the pairwise discrimination task relevant to deal screening and capital allocation.

This binary framing is consistent with established practice in the M&A ML literature @zhang2024 @ajayi2022 and aligns the prediction target with what the feature space can realistically support at the available sample size.

#line(length: 100%, stroke: 0.4pt + gray)

== Synthesis: Gap Table and Research Hypotheses

The preceding four streams establish a clear landscape: each prior paradigm captures one dimension of synergy potential while structurally discarding the others. The table below maps each prior approach to its specific failure mode and the architectural response this study deploys.

#figure(
  caption: [
    Synthesis of Prior Literature: Structural Failure Modes
    and Research Gaps Addressed by This Study
  ],
  table(
    columns: (1.7fr, 1.4fr, 2.0fr, 1.9fr),
    align: (left, left, left, left),
    inset: 6pt,
    stroke: 0.4pt,
    table.header(
      [*Prior Work*], [*Method*], [*Structural Failure*],
      [*This Study's Response*]
    ),
    [Palepu (1986); Barnes (1990)],
    [Logit / MDA on ratios],
    [Linearity bias; pseudo-R² < 0.10; treats firms as i.i.d. points],
    [Non-linear XGBoost fusion with graph topology; treats firms as network nodes],

    [Zhang et al. (2024)],
    [Random forest / XGBoost on financial ratios],
    [Independence assumption; survivorship bias; look-ahead contamination; no network features],
    [Temporal train/test splits; GraphSAGE neighbourhood aggregation; full multimodal fusion],

    [Elhoseny et al. (2022)],
    [Deep MLP on financial distress],
    [High accuracy on distress; wrong target; no relational structure],
    [CAR direction as target; relational graph operator; cross-document text similarity],

    [Zhao et al. (2020); Han et al. (2023)],
    [BERT / RoBERTa → scalar sentiment],
    [Tabular trap: embeddings flattened to single score; predicts deal occurrence not outcome],
    [FinBERT on section-split 10-Ks; binary CAR classification; pairwise cosine similarity],

    [Hajek et al. (2024)],
    [FinBERT news sentiment],
    [Single-document; no acquirer-target pairwise delta; wrong prediction target],
    [Pairwise acquirer-target cosine distance; section-specific semantic divergence (H2)],

    [Loughran & McDonald (2011); Hoberg & Phillips (2016)],
    [BoW / TF-IDF similarity],
    [Compositional blindness; deal occurrence target; no outcome prediction],
    [Contextual FinBERT embeddings; section-specific similarity scores],

    [Cohen & Frazzini (2008); Ahern & Harford (2014)],
    [Industry-level supply chain analysis],
    [Industry-level only; no firm-level GNN encoding; no CAR prediction],
    [Firm-level GraphSAGE on Bloomberg SPLC data; CAR direction as target (H1, H3)],

    [Venuti (2021)],
    [Homogeneous GraphSAGE],
    [Predicts deal likelihood; homogeneous edges; no text or financial fusion],
    [Heterogeneous edge types (supplier, customer, competitor); full multimodal late fusion],

    [Baltrušaitis et al. (2019); Xu et al. (2021)],
    [Multimodal fusion frameworks],
    [Not applied to M&A synergy outcome; no graph modality; no ablation design],
    [Full Block A+B+C late fusion with XGBoost; controlled ablation ladder; SHAP decomposition],
  )
) <tab:synthesis>

=== Formal Research Hypotheses

The following three hypotheses emerge from the synthesis above. Each is operationalised to map precisely to an ablation experiment in the empirical design.

*H1 — The Topological Alpha Hypothesis:* The inclusion of supply chain and competitor network topology (via GraphSAGE) will yield a statistically significant increase in AUC-ROC ($p < 0.05$, paired $t$-test) relative to the financial-only baseline under 5-fold stratified cross-validation. This gain will be disproportionately concentrated within supply-chain-dependent manufacturing sectors (SIC 20–49) compared to asset-light service sectors (SIC 60–79), reflecting the hypothesis that graph embeddings recover signal proportional to the structural density of the industrial ecosystem. The paired $t$-test is used because the comparison is between two models evaluated on the same cross-validation folds, producing naturally paired observations that control for fold-level difficulty variation.

*H2 — The Semantic Divergence Hypothesis:* The predictive relationship between textual similarity and synergy direction is *conditional* on the document section. High cosine similarity in strategic disclosures (MD&A) will positively correlate with $"CAR"$ — reflecting strategic fit — whereas high similarity in Risk Factors will negatively correlate with $"CAR"$ — reflecting risk concentration. This conditional directionality is tested via bivariate OLS regression: $C A R_i = beta_0 + beta_1 dot "sim"_i^("MDA") + beta_2 dot "sim"_i^("RF") + epsilon_i$. The bivariate specification is critical: firms with high MD&A similarity often also exhibit high Risk Factor similarity, and only a joint model can isolate their independent effects. The sign asymmetry — $beta_1 > 0$ and $beta_2 < 0$ — directly refutes the monotonic sentiment utility assumption embedded in standard NLP classifiers.

*H3 — The Topological Arbitrage Hypothesis:* Acquirer nodes exhibiting high betweenness centrality in the heterogeneous supply chain graph will exhibit statistically compressed variance in $|"CAR"|$ outcomes relative to peripheral nodes, as measured by Levene's test for equality of variance across betweenness centrality quantile groups. This compression reflects the bilateral dependency constraints of bridge nodes: their network position creates both upside synergy capture opportunities and downside disruption exposure, and the net effect is a dampening of extreme outcomes in either direction. Levene's test is chosen over Bartlett's test because $|"CAR"|$ is right-skewed with heavy tails characteristic of financial return data, and Levene's test does not assume normality of the underlying distribution — a critical property for financial return data with the skewness and kurtosis observed in this sample.

#bibliography("works-litreview.bib", style: "ieee")