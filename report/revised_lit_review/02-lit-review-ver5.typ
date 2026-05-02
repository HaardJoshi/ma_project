// ============================================================
//  02-literature-review.typ  (Polished — v2)
//  Chapter 2: Literature Review
//  M&A Synergy Prediction | Hard Joshi | UEL
// ============================================================

= Literature Review

== The Central Argument: A Gap Three Decades Wide

Every year, thousands of companies announce mergers and acquisitions — deals in which one firm pays a premium to absorb another, betting that the combined entity will be worth more than the sum of its parts. Yet decades of evidence suggest that this bet fails more often than it succeeds. The central question motivating this study is simple: if this failure rate has persisted across a century of increasingly sophisticated corporate due diligence, what information are existing analytical frameworks systematically missing?

This review builds a sustained argument across four interconnected knowledge streams, each representing a distinct wave of scholarship that has tried — and partially failed — to answer that question. Rather than a neutral survey, each stream is examined for what it structurally could not achieve, with each identified gap mapped directly to one of the three hypotheses tested in this dissertation.

+ *Stream I — The M&A Paradox and the Failure of Valuation Theory:* diagnoses why synergy prediction is genuinely difficult and establishes that the root cause is not a lack of computing power, but a structural information blindspot.

+ *Stream II — The Tabular Paradigm:* traces the full history of quantitative M&A models — from logistic regression through gradient-boosted trees and early deep neural networks — and demonstrates that every generation inherited the same core architectural limitation, producing an asymptotic accuracy ceiling no amount of algorithmic sophistication could break through.

+ *Stream III — The Semantic Turn:* examines how natural language processing entered the M&A prediction landscape and why, despite genuinely powerful tools, the field systematically pointed them at the wrong question — and even when pointed correctly, fell into a structural trap that cancelled the very richness it sought to exploit.

+ *Stream IV — The Topological Turn:* engages with supply chain finance, corporate network theory, and graph neural network research to establish that the structure of a firm's external relationships — its suppliers, customers, and competitors — carries economic information about deal value that is, by mathematical necessity, invisible to any model that treats each firm as an isolated data point.

The review concludes that these four streams converge on a single architectural response: a heterogeneous graph model that simultaneously fuses financial fundamentals (Block A), section-conditioned textual embeddings (Block B), and supply-chain and competition graph topology (Block C). As of 2025, no published study has directed this combined architecture at post-merger synergy outcome prediction.

== Stream I: The M&A Paradox and the Failure of Valuation Theory

=== The Empirical Record: Why Deals Keep Failing

Mergers and acquisitions are the primary mechanism through which capital is reallocated across the global economy. When a company acquires another, the underlying premise is that the combined entity will generate more value than the two firms could independently — through cost savings, revenue growth, market power, or access to new capabilities. This theoretical surplus is called "synergy."

The empirical record, however, tells a more troubling story. @martynova2008 documented that between 70% and 90% of acquisitions fail to generate value for the acquirer's shareholders — a finding replicated across multiple independent research programmes @christensen2011. More striking than the magnitude of this failure rate is its persistence: it has remained statistically stable across entirely different economic eras, from the conglomerate wave of the 1960s through the technology boom of the late 1990s and the post-financial-crisis consolidation wave of the 2010s @martynova2008. A failure rate that survives such varied conditions is unlikely to be cyclical; it points toward something structurally wrong with how deals are evaluated.

@bradley1988 established a revealing asymmetry in the data: while shareholders of acquired firms (the targets) captured average announcement-day abnormal returns of approximately 30%, the shareholders of acquiring firms systematically lost value, with combined deal synergy gains averaging only 7.4% of combined pre-deal firm value. The gap between what acquiring firms paid and what the market believed the combination was worth is precisely what this study attempts to predict.

=== Behavioural Distortions: When Executive Confidence Contaminates the Data

The first layer of explanation for persistent M&A failure is behavioural. @roll1986 proposed the Hubris Hypothesis: acquiring managers systematically overestimate their own ability to extract value from a target, treating the market's price for that target as an underestimate rather than as a rational, information-weighted signal. This overconfidence drives them to pay premiums that the actual post-merger performance cannot justify — what @sirower1997 termed the "Synergy Trap."

From a machine learning perspective, this creates a specific and irreducible data quality problem. If the acquisition premiums recorded in deal databases partly reflect executive overconfidence rather than genuine synergy potential, then any model trained to predict outcomes from those premiums is fitting a signal that has been systematically contaminated with human behavioural error. No amount of regularisation or hyperparameter tuning can remove bias that is baked into the target variable itself. This is one of the reasons this study uses Cumulative Abnormal Return (CAR) — the stock market's immediate revision of its estimate of firm value upon hearing the deal announcement — as the prediction target, rather than deal premia or accounting-based synergy measures. CAR represents the market's collective, forward-looking assessment of deal value and is considerably less susceptible to the narrative distortions of individual executives @mackinlay1997.

=== Market Timing: When Deal Waves Are About Valuation, Not Strategy

A complementary source of noise in M&A data was identified by @shleifer2003. Their stock-market-driven acquisitions model demonstrates that companies with temporarily overvalued equity have a rational incentive to use their inflated shares as acquisition currency — essentially exchanging overpriced paper for real assets. Under this model, a substantial fraction of any M&A wave is driven by relative misvaluation rather than genuine operational complementarity between the two firms. The observed announcement-period returns, in these cases, reflect markets correcting for overvaluation rather than pricing synergy realisation. @grossman1980 established the theoretical boundary here: in a world where gathering and processing information is costly, markets cannot be perfectly efficient, and persistent information asymmetries around deal announcements are a natural and stable equilibrium. This bounded-efficiency framing is what makes the prediction problem tractable in the first place — if markets perfectly and instantaneously priced all available information, there would be no learnable signal left.

=== The Epistemological Barrier: What Due Diligence Cannot See

Even setting aside behavioural and timing distortions, the fundamental problem is that conventional due diligence is architecturally blind to certain categories of risk. @angwin2001 documented that standard due diligence operates in organisational silos: financial, legal, and operational teams independently assess their respective domains without modelling how those domains interact. This fragmentation means that multiplicative risks are systematically missed. Consider a firm with strong cash flows that happens to source 80% of its critical components from a single supplier. Each dimension passes a standard due diligence check in isolation; the conjunction — that the entire cash-flow advantage is conditional on a single supply relationship — is invisible to teams that never share information.

@akerlof1970 provided the theoretical foundation for why mandated disclosures cannot fully resolve this problem. His "Market for Lemons" framework shows that when buyers and sellers have asymmetric information, sellers have an incentive to obscure their vulnerabilities within the noise of standard reporting formats. A 10-K filing, however detailed, is ultimately a document a company prepares about itself. What it cannot obscure — at least not easily — is the verifiable structure of its external relationships: which firms it buys from, sells to, and competes with. This network of real economic ties is precisely what Block C in this study's architecture encodes, and it is the one source of synergy-relevant information that is most resistant to strategic misrepresentation.

_Stream I conclusion:_ The M&A failure rate is partly predictable from structural information that existing valuation frameworks systematically exclude. The following three streams each diagnose a specific category of excluded information and the architectural limitations that kept it out.

== Stream II: The Tabular Paradigm — An Asymptotic Ceiling

=== The Econometric Era: When Linearity Assumed Too Much

Before examining the historical progression of quantitative M&A models, the structural limitation of the "tabular" paradigm must be defined. A tabular model takes a spreadsheet-style matrix of numbers — one row per deal, one column per financial ratio or deal characteristic — and learns statistical patterns that separate value-creating deals from value-destroying ones. Each deal is treated as an independent, self-contained data point. The model has no mechanism for knowing that two deals involve acquirers who share a supplier, or that the target in one deal is a customer of the acquirer in another. It sees the numbers; it does not see the relationships.

The first generation of quantitative M&A models — most influentially @palepu1986 and @barnes1990 — used logistic regression and multiple discriminant analysis on financial ratios. These approaches assume that each financial variable has a monotonic, linear relationship with the outcome: more of a given ratio always either increases or decreases the probability of a successful deal. @barnes1990 demonstrated empirically that this assumption breaks down for leverage. Moderate debt signals fiscal discipline and managerial confidence; very high debt signals financial fragility. The relationship is non-monotonic — it changes direction at some threshold — and linear models fundamentally cannot represent direction changes.

More revealing still, @palepu1986 own out-of-sample validation found that acquisition targets could not be predicted reliably beyond chance levels. This was not a data collection problem or an implementation flaw; it was evidence that the information contained in financial ratios was simply insufficient to reconstruct synergy outcomes. The consistently low pseudo-R² values (below 0.10) documented across econometric studies @betton2008 are not calibration failures — they are a measurement of the information gap that financial ratios leave behind. This ceiling is the first empirical benchmark any successor model must demonstrably improve upon.

=== Why Machine Learning Could Not Fully Escape the Same Trap

The arrival of ensemble machine learning methods — random forests, gradient-boosted decision trees, and ultimately XGBoost — promised to resolve the non-linearity problem. @zhang2024 and similar studies deployed these techniques on financial ratio vectors and did report accuracy improvements over logistic baselines. Those improvements are real, but they are bounded by four structural problems that persist regardless of how sophisticated the algorithm becomes.

*The Independence Ceiling:* Both random forests and XGBoost operate on rows of a feature table where each row represents one deal in complete isolation. The algorithm learns patterns within individual deals but has no mechanism to propagate information between related deals. If two acquiring firms share a common supplier, and that supplier is experiencing financial distress, both deals carry a latent risk that is invisible in their individual financial ratios. This is not a data-engineering problem that can be solved by adding more features — it is an architectural impossibility for any model that processes deals independently @hamilton2017.

*Survivorship Bias:* M&A datasets are almost exclusively drawn from completed transactions. Deals that were considered but abandoned — often precisely because due diligence revealed serious problems — never appear in the training data. This systematically biases estimated model coefficients toward the characteristics of deals that proceeded, rather than deals that actually created value @betton2008. No machine learning algorithm can correct for data that was never collected.

*Look-Ahead Contamination:* Several prominent ML studies, including @zhang2024, construct financial features from annual filings without explicitly verifying that the fiscal year end-date precedes the deal announcement date. When a model trains on financial data that postdates the deal it is predicting, its apparent accuracy reflects information that would not have been available at the time of the decision — inflating reported performance beyond any economically realisable level.

*Crystallised Analyst Bias:* A common preprocessing step in these pipelines is discretising continuous financial variables into categorical bins before feeding them into tree models. This encodes whatever assumptions the analyst made when deciding where to draw the bin boundaries into the feature representation in a way that cannot be undone. The model then learns from a filtered, pre-interpreted version of the data rather than the raw signal — an automated version of the same heuristic reasoning employed by human analysts whose systematic failures are the very motivation for building quantitative models in the first place.

Taken together, these four problems create what can be described as an asymptotic accuracy ceiling: once a tabular model has captured all the linear, rank-order, and non-parametric signal available in deal-level financial features, no further improvement is achievable by making the model more sophisticated. The ceiling is a function of the information content of the feature space, not of computational power.

=== Why Deep Neural Networks Did Not Break Through

The introduction of deep multi-layer perceptrons (MLPs) and recurrent sequence models appeared to dissolve the linearity constraint entirely. @elhoseny2022 reported 95.8% accuracy on financial distress prediction using a deeply optimised neural network — a figure that superficially suggests the prediction problem has been solved. It has not, for a reason that is worth spelling out carefully.

Financial distress prediction and M&A synergy prediction are fundamentally different questions. The former asks: "Is this firm, on its own, likely to fail?" That question can be answered, at least in part, from a single firm's own financial time series. The latter asks: "Will combining these two specific firms create more value than they possess separately?" That is an inherently relational question — its answer depends on the interaction between the two firms and on their positions within a broader industrial ecosystem. An MLP with ten layers and ten thousand parameters, trained on a vector of fifty financial ratios, still operates in a fifty-dimensional space that is topologically flat. It has no representation of whether the acquirer's largest customer is about to go bankrupt, or whether the target and acquirer share a fragile single-source supplier. These omissions cannot be corrected by adding more layers to the same architecture.

_Stream II conclusion:_ Every generation of tabular model — logistic regression, random forest, XGBoost, MLP, LSTM — inherits the independence assumption from its predecessor. The asymptotic ceiling is not a function of model complexity but of the information content of the feature space. Breaking through it requires a fundamentally different kind of model: one that operates on relationships rather than rows. Stream IV addresses this. Before arriving there, Stream III examines why adding language data — an intuitively promising additional information source — proved far more complicated than the field anticipated.

== Stream III: The Semantic Turn — Powerful Tools, Wrong Targets

=== The Bag-of-Words Era: A Useful Foundation with Hard Limits

Where tabular models read financial numbers, semantic models read text — in the M&A context, primarily the annual 10-K reports that US-listed firms file with the Securities and Exchange Commission. These filings contain structured, standardised sections: the Management Discussion and Analysis (MD&A), in which management describes strategy, performance, and outlook in their own words, and the Risk Factors section, in which the company enumerates specific threats and uncertainties it faces. The intuition motivating semantic M&A research is that these disclosures might reveal complementarity or conflict between acquirer and target that financial ratios alone cannot capture.

The foundational contribution to financial NLP is @loughran2011, who demonstrated that standard general-purpose sentiment dictionaries systematically mislabel financial language. Words like "liability," "risk," and "obligation" carry negative connotations in ordinary English but are routine, neutral legal descriptors in financial filings. Their domain-specific word lists corrected this systematic mislabelling and established a principle that remains foundational: financial text requires tools calibrated for the specific conventions of financial language. This study directly inherits that principle in its choice of FinBERT over general-purpose language models.

The bag-of-words (BoW) paradigm that dominated early financial NLP, however, carries a structural limitation that no amount of vocabulary refinement can overcome: it is compositionally blind. A BoW model represents a document as a count of how many times each word appears, with no memory of the order or context in which words appeared. The sentence "We have eliminated our exposure to risk" and the sentence "We have significant exposure to risk" produce nearly identical word-frequency vectors while conveying directly opposite strategic signals. In M&A filings, where a single conditional clause — "subject to regulatory approval" — can invert the economic meaning of a multi-billion-dollar commitment, this ambiguity is not a minor inconvenience; it is a systematic failure mode.

Despite this, a substantial body of recent research has continued to extend the BoW approach: @demers2024 construct frequency-based lexicons for human capital disclosures; @acheampong2025 proxy financial constraints using lexicon-based indices; @garcia2020 develop finance-specific dictionaries validated through market data. Each of these contributions improves the vocabulary but cannot address the underlying compositional blindness. @hoberg2016 represents the most structurally sophisticated BoW application — their text-based network industries classification used product-description cosine similarity from 10-K filings to construct time-varying firm similarity networks, demonstrating that textual proximity predicts competitive dynamics and deal likelihood. This study inherits that semantic proximity methodology directly, but redirects it from predicting whether a deal will happen to predicting whether a deal that has happened will create or destroy value, and replaces TF-IDF word vectors with contextualised FinBERT embeddings.

=== The Transformer Revolution: A Better Tool, But Often Misdirected

@devlin2018 BERT architecture marked a genuine qualitative advance in language modelling. Rather than counting word occurrences, BERT learns contextual embeddings — vector representations in which the same word receives a different numerical encoding depending on the surrounding words. This directly resolves the compositional blindness of BoW models. @araci2019 FinBERT specialised this architecture for financial language through pre-training on a large corpus of corporate filings, giving it domain-calibrated representations that a general BERT model lacks.

The challenge is not with the tools themselves but with how the field has applied them to M&A prediction. @zhao2020 and @han2023 both used transformer architectures to extract semantic features from M&A-related text, but then compressed the resulting rich, high-dimensional representations into a single scalar sentiment score before feeding them into a downstream classifier. This pipeline — transformer as sophisticated word counter, followed by scalar compression — preserves almost none of the representational richness that motivated using a transformer in the first place. The architecture is, in terms of information content, not substantially different from an advanced bag-of-words approach.

More fundamentally, @hajek2024 and @han2023 both direct their NLP pipelines at predicting *deal occurrence* — whether an acquisition announcement will happen — rather than *deal outcome* — whether the announced deal will create or destroy shareholder value. These are genuinely different problems. Predicting deal occurrence requires detecting patterns in news flow that precede announcements; predicting deal outcome requires modelling the fundamental complementarity between acquirer and target across multiple dimensions. The existing NLP literature has made substantial progress on the former and has largely not addressed the latter.

=== Why Standard FinBERT Cannot Fully Recover the Textual Signal

Even correctly directed at deal outcome prediction, the standard single-document FinBERT implementation faces three structural limitations that this study directly addresses.

*Frozen generic embeddings:* FinBERT's pre-training objective is general financial language modelling — predicting masked words in financial text — not the specific task of discriminating strategic alignment from risk concentration. Its frozen weights cannot adapt to the semantic distinctions most relevant to synergy prediction. This study addresses this by treating the two dominant 10-K sections — MD&A and Risk Factors — as semantically distinct modalities rather than interchangeable text, exploiting a distinction FinBERT cannot make on its own.

*Single-document architecture:* Studies that apply FinBERT independently to the acquirer's filing or the target's filing separately miss the cross-document signal that matters most for synergy prediction. It is not the absolute strategic sophistication of either party that predicts synergy — it is the *relative alignment* between the two, computed as the cosine distance between paired acquirer-target embeddings from the same document section. This pairwise computation is the core of H2 in this study.

*Section conflation:* Standard implementations embed entire 10-K documents or undifferentiated excerpts, mixing two sections that this study's H2 predicts have *opposite* relationships with deal value. MD&A similarity (shared strategic direction) is hypothesised to correlate positively with announcement returns; Risk Factor similarity (overlapping liability exposure) is hypothesised to correlate negatively. A model that conflates these sections fits a noisy mixture of opposing signals, systematically suppressing the predictive coefficient of each. The −0.012 AUC drop observed when undifferentiated text embeddings are added to the model (M2 vs. M1) is the empirical consequence of exactly this conflation.

The choice of FinBERT over longer-context alternatives such as Longformer is justified on three grounds: domain-specific vocabulary alignment with 10-K filing language; the section-specific extraction protocol used here, which makes the 512-token context window a non-binding constraint in practice; and empirical evidence from @qin2019 that frozen domain-specific representations with lightweight downstream prediction heads perform competitively on financial classification tasks at sample sizes comparable to this study's deal universe.

_Stream III conclusion:_ Transformer-based NLP has the representational capacity to capture strategic fit signals in 10-K filings. That capacity has not been applied to post-merger synergy outcome classification. Furthermore, even when correctly directed, text models operating on isolated documents miss the second-order structural reality: two firms with perfectly aligned strategies can still fail to create synergy if their network positions make the combined entity more fragile. This structural incompleteness is what Stream IV addresses.

== Stream IV: The Topological Turn — Network Structure as Irreducible Signal

=== The Economic Foundation: Supply Chain Momentum

Before examining the empirical evidence, it is worth clarifying what "network topology" means in this context. Modern corporations do not exist in isolation — they are embedded in dense webs of economic relationships. A manufacturer buys components from dozens of suppliers; a retailer sells through hundreds of partners; competitors share customers, regulatory exposure, and talent pools. These relationships create dependencies: if a key supplier fails, the customer's production line can stop; if a competitor is acquired and strengthens their market position, the remaining firm's pricing power erodes. Topology refers to the *structure* of these relationship networks — not just which firms are connected, but how centrally positioned each firm is, how many paths between other firms run through it, and whether it serves as a critical bridge between otherwise disconnected clusters. The central hypothesis of Stream IV is that this structural position encodes information about how a deal will be received by the market — information that balance sheets and management narratives cannot convey.

The empirical case for this hypothesis is grounded in a well-established finding from financial economics rather than in machine learning novelty. @cohen2008 demonstrated that economic shocks to a supplier do not immediately price into the stock of the supplier's customer. Due to information friction and limited investor attention, these shocks propagate across supply chain links with a measurable time lag, generating a predictable return momentum effect for investors who monitor inter-firm dependency structures. This "supply chain momentum" phenomenon is direct empirical evidence that network topology encodes information about future firm value that is absent from standalone financial metrics — and that this information persists long enough to be exploited, precisely because most models process firms in isolation.

@ahern2014 extended this reasoning directly to M&A. They demonstrated that the structure of industry-level trade networks predicts the direction of merger waves: acquisitions cluster along supply chain linkages because acquiring firms seek to internalise relationships that generate the highest dependency-reduction value. More directly relevant to this study, they found that supply chain proximity at the industry level predicts post-merger combined stock returns — providing precedent that network position carries synergy-relevant information over and above financial fundamentals. This study operationalises the same insight at the firm level using Bloomberg SPLC data, enabling node-level rather than industry-level topology encoding.

=== Corporate Network Centrality and Value

The broader corporate networks literature reinforces this foundation. @fee2004 documented that customer-supplier relationships carry systematic pricing power information: firms with high customer concentration earn measurably different returns upon resolution of those relationships, confirming that vertical network dependencies have real equity value implications. @larcker2013 demonstrated that director network centrality — specifically betweenness centrality in board interlock networks — is a significant predictor of future firm performance and acquisition premia, establishing precedent that graph-theoretic centrality measures carry economic signal in corporate finance contexts.

To understand betweenness centrality intuitively: imagine a social network where some people are highly connected to many different groups while others only know people within their own immediate circle. Betweenness centrality measures how often a given node sits on the shortest path between two other nodes — in other words, how much of the network's "traffic" must pass through it. A firm with high betweenness centrality in the supply-chain-competitor graph is a structural broker: many supplier-customer or competitor-competitor connections in the industrial ecosystem run through it. These "bridge nodes" face a distinctive dual constraint: their centrality creates both upside synergy capture opportunities (they can internalise strategically important relationships) and downside disruption exposure (shocks from any connected cluster propagate to them quickly). This bilateral constraint is the mechanism hypothesised in H3 to compress the variance of announcement-period returns relative to peripheral acquirers.

=== Why Adding Centrality to XGBoost Is Insufficient

A sophisticated critic might ask: if betweenness centrality is valuable information, why not simply compute it and add it as an additional column in the feature table? This scalar centralisation approach has two structural limitations that motivate a full graph neural network architecture.

First, scalar centrality measures collapse a high-dimensional structural signal into a single number, discarding the specific *pattern* of connectivity that carries economic meaning. Two firms can share identical betweenness centrality scores while occupying radically different network contexts: one may bridge two healthy, high-growth industrial clusters; the other may bridge two sectors in secular decline. A scalar feature treats these as identical; a graph embedding that propagates and aggregates neighbourhood information preserves the distinction @hamilton2017.

Second, scalar features cannot represent second-order network effects — dependencies that run through intermediate nodes. If an acquirer's primary supplier is itself a customer of the target, this creates a circular dependency loop that substantially changes the risk profile of the combined entity. This loop is invisible to any feature engineering pipeline that processes each deal in isolation. A message-passing graph neural network propagates information through multi-hop neighbourhoods by design, recovering these higher-order structural patterns without requiring their explicit enumeration @hamilton2017.

@venuti2021 provided practical validation of this argument: GraphSAGE achieved 81.79% accuracy in predicting acquisitions for private companies where financial data is sparse — precisely the regime where structural neighbourhood information provides the largest marginal value over standalone features. This establishes the inductive architecture's applicability to data-scarce M&A contexts.

=== Why Edge-Type Semantics Matter: The Case for Heterogeneous Graphs

A final refinement is necessary before the graph architecture can be considered adequate. A homogeneous graph model — one that treats all inter-firm edges as equivalent — commits a category error that compromises its predictions.

A `supplies_to` relationship implies operational dependency and risk propagation: if the supplier fails, the customer's production line stops. A `competes_with` relationship implies market concentration and pricing power dynamics: if two competitors merge, their combined market share may attract regulatory intervention and change the competitive landscape for all remaining players. These are fundamentally different economic mechanisms. Collapsing them into a single undirected edge type introduces a mixed signal that obscures what each relationship type independently encodes — analogous to averaging the coefficients of variables that have opposite effects.

@shi2017 formalised the theoretical basis for Heterogeneous Information Networks (HINs) and demonstrated that type-specific attention mechanisms during neighbourhood aggregation consistently outperform homogeneous baselines on multi-relational graphs. @wang2021 extended this into the Heterogeneous Graph Attention Network (HAN), showing that relationship-type-specific aggregation recovers semantic structure that single-type models discard. @lv2021 provided systematic benchmarks confirming heterogeneous graph encoders achieve statistically superior performance across multiple classification tasks on multi-relational corporate networks.

This study constructs a *heterogeneous* graph $cal(G) = (cal(V), cal(E), cal(T)_v, cal(T)_e)$ with semantically distinct edge encodings for `supplier_of`, `customer_of`, `competitor_of`, and `acquires` relationships within PyTorch Geometric's HeteroData structure. Type-specific GraphSAGE aggregation functions are applied independently per relationship type before cross-type pooling — recovering exactly the semantic distinctions that scalar feature engineering and homogeneous graph models discard.

_Stream IV conclusion:_ Graph neural network architectures are theoretically well-suited to encoding inter-firm dependencies, and supply chain network structure is empirically demonstrated to contain synergy-relevant information. Yet no published study has directed a heterogeneous GNN at post-merger synergy *outcome* classification using firm-level topology. The Topological Alpha Hypothesis (H1), the Topological Arbitrage Hypothesis (H3), and the HeteroGraphSAGE architecture proposed in this study constitute the original contribution at this intersection.

== The Measurement Foundation: Event Study Methodology

=== How Announcement Returns Are Measured

While Streams I through IV establish the feature categories required to *predict* synergy, a rigorous framework is required to *measure* synergy itself — a non-trivial problem given that the true economic benefits of a merger may take years to fully materialise. The standard solution in empirical corporate finance, formalised by @mackinlay1997 and whose statistical properties were rigorously characterised by @brownwarner1985, is the event study.

The core idea is elegant: if financial markets are reasonably efficient at incorporating publicly available information, then the stock price reaction to a deal announcement reflects the market's best estimate of the present value of future synergies. A large positive price reaction means the market believes the deal will create substantial value; a negative reaction means it will destroy value.

The methodology operationalises this through a market model. For each acquirer $i$, a "normal" return generating process is estimated over a pre-announcement estimation window:

$ R_(i,t) = alpha_i + beta_i R_(m,t) + epsilon_(i,t) $

where $R_(i,t)$ is the firm's daily return, $R_(m,t)$ is the broad market return, and $alpha_i$ and $beta_i$ are firm-specific parameters. Abnormal Returns (ARs) during the announcement period are then computed as the difference between what the firm actually earned and what the model predicted it should have earned given market conditions:

$ A R_(i,t) = R_(i,t) - (hat(alpha)_i + hat(beta)_i R_(m,t)) $
The Cumulative Abnormal Return ("CAR") sums these daily excess returns across the event window:

$ "CAR"_i = sum_(t=t_1)^(t_2) A R_(i,t) $
A positive CAR indicates the market interpreted the deal as value-creating; a negative CAR indicates value destruction. This is the binary outcome label — positive or negative — that all models in this study attempt to predict.

=== Event Window Choice

@mackinlay1997 established that event window selection involves a fundamental bias-variance trade-off. Narrow windows (e.g., $[-1, +1]$ trading days) isolate the immediate announcement surprise cleanly but risk missing delayed market reactions for deals where information leaked before the formal announcement. Wider windows (e.g., $[-5, +5]$) capture more complete price discovery but introduce additional variance from unrelated news.

This study uses the $[-5, +5]$ window, consistent with the empirical M&A literature @betton2008, as a practical balance that accommodates the pre-announcement information leakage documented to begin three to five days before announcement in approximately 25% of transactions. Robustness checks under the narrower $[-1, +1]$ specification confirm that the core directional findings are not window-specific.

=== Why Binary Classification Rather Than Regression

The signal-to-noise characteristics of short-window CARs make predicting their *magnitude* a challenging target for machine learning models. Announcement return magnitudes are driven by surprise — and surprise, by definition, is hard to predict from publicly available information. The consistently low pseudo-R² values documented in the literature @palepu1986 @betton2008 are consistent with the Efficient Markets Hypothesis's implication that the *size* of the market's reaction to new information is largely unpredictable.

The *direction* of the reaction, however, contains learnable signal rooted in deal fundamentals: whether the acquirer has the financial capacity to absorb the target, whether their strategies are complementary, and whether their ecosystem positions make the combination structurally sound. This study therefore defines $y_i = 1$ if $"CAR"_i > 0$ (value-creating) and $y_i = 0$ otherwise, with AUC-ROC as the primary evaluation metric for its robustness to class imbalance and threshold-invariant characterisation of discriminative power @ajayi2022.

== The Multimodal Imperative: Why Fusion Is the Only Exit

=== Complementary Variance: Three Lenses on the Same Deal

The four streams surveyed above converge on a structural finding: financial, textual, and topological features encode different, partially non-overlapping dimensions of synergy potential. @baltrusaitis2019 formalize this as "complementary variance" in their taxonomy of multimodal learning architectures. A firm may exhibit strong capital adequacy (visible in financial ratios), precise strategic alignment with its target (visible in text), yet suffer from fragile single-source supplier dependencies (visible only in graph topology). A model that observes only one of these dimensions is mathematically blind to whatever divergences exist in the others, and no within-modality engineering can recover information that was never measured.

@xu2021 demonstrated the practical validity of this in a financial forecasting context: structured financial data and unstructured text features carry statistically independent predictive signal — signal that vanishes from each modality when the other is controlled for, but is recoverable when both are jointly modelled. @qin2019 demonstrated the same complementarity between price-based and news-based features for stock movement prediction.

=== Why HeteroGraphSAGE Fusion Is the Logical Response

Standard baselines — logistic regression on financial ratios, XGBoost on financial ratio vectors, standalone FinBERT applied to individual documents, homogeneous GCN with scalar centrality features — each fail for a specific, architecturally irreversible reason. @tbl-gap maps each study to its structural failure mode and the component of this study's architecture designed to address it.

#figure(
  table(
    columns: (1.5fr, 1.5fr, 2fr, 2fr),
    align: (left, left, left, left),
    stroke: (x, y) => if y == 0 { (bottom: 1pt + black, top: 1pt + black) } else { none },
    inset: 8pt,

    table.header(
      [*Literature Stream*], [*Legacy Method*], [*Structural Failure (The Ceiling)*], [*Implemented Architecture (The Exit)*],
    ),

    // STREAM II: TABULAR PARADIGM
    table.cell(colspan: 4, fill: luma(240))[*Stream II: The Tabular Paradigm*],
    [Palepu (1986); Barnes (1990)], [Logit / MDA on ratios], [Linearity bias; pseudo-R² < 0.10; no network features], [Non-linear fusion engine; Blocks A+B+C],
    [Zhang et al. (2024)], [XGBoost / Random Forest], [i.i.d. assumption; survivorship bias; look-ahead contamination], [GraphSAGE neighbourhood aggregation; temporal splits],
    [Elhoseny et al. (2022)], [Deep MLP (AWOA-DL)], [Topologically flat; predicts financial distress, not synergy], [Relational graph operator; synergy CAR as target],

    // STREAM III: SEMANTIC TURN
    table.cell(colspan: 4, fill: luma(240))[*Stream III: The Semantic Turn*],
    [Loughran & McDonald (2011)], [Bag-of-Words (BoW)], [Compositional blindness; deal occurrence target], [Contextual FinBERT embeddings; section-specific similarity (H2)],
    [Hajek et al. (2024)], [FinBERT sentiment], [Single-document; no pairwise delta; acquisition likelihood target], [Pairwise acquirer-target cosine distance; H2 conditional directionality],
    [Zhao et al. (2020); Han et al. (2023)], [BERT / RoBERTa → XGBoost], [Conflates strategic and risk disclosures; predicts occurrence not outcome], [Section-specific semantic splitting; binary CAR classification (H2)],

    // STREAM IV: TOPOLOGY AND MULTIMODAL FUSION
    table.cell(colspan: 4, fill: luma(240))[*Stream IV: Topology and Multimodal Fusion*],
    [Cohen & Frazzini (2008); Ahern & Harford (2014)], [Industry-level supply chain analysis], [Industry-level only; no firm-level GNN; no CAR prediction], [Firm-level GraphSAGE on SPLC data (H1, H3)],
    [Venuti (2021)], [Homogeneous GraphSAGE], [Collapses edge semantics (suppliers = competitors); deal likelihood target], [Heterogeneous edge types; full multimodal fusion],
    [Baltrušaitis et al. (2019); Xu et al. (2021)], [Multimodal fusion frameworks], [Not applied to M&A synergy; no graph modality], [Block A+B+C late fusion with SHAP decomposition],
  ),
  caption: [Methodological Evolution Matrix: structural failures of prior work and architectural responses.],
) <tbl-gap>

The HeteroGraphSAGE fusion architecture resolves each limitation simultaneously: GraphSAGE's inductive neighbourhood aggregation recovers the topological signal that tabular models cannot access; FinBERT's contextual embeddings on section-split 10-K filings recover the semantic signal that BoW and scalar NLP cannot represent; late fusion via concatenation preserves modality-specific representations while enabling cross-modal learning in the joint prediction head; and heterogeneous edge-type encoding preserves the semantic distinctions between relationship types that homogeneous graph models discard.

=== Architectural Choice: Late Fusion and Its Rationale

The M&A deal universe with complete multimodal coverage is approximately 2,800–5,000 observations — substantially below the sample sizes typically required for stable joint end-to-end training of transformer and GNN components @baltrusaitis2019. This study therefore adopts late fusion: each modality is encoded independently into a fixed-dimensional embedding vector ($bold(h)_F in RR^(d_F)$, $bold(h)_T in RR^(d_T)$, $bold(h)_G in RR^(d_G)$), and these vectors are concatenated into a joint representation $bold(z)_i = [bold(h)_F || bold(h)_T || bold(h)_G]$ before a shared prediction head. This design isolates modality-specific representation learning from cross-modal inference, enabling robust individual stream training even when certain modalities have incomplete coverage.

The downstream prediction head uses @chen2016 XGBoost framework for the baseline ablation experiments, given its well-documented advantages over neural networks on heterogeneous tabular feature vectors at M&A sample sizes. SHAP decomposition @lundberg2017 is applied post-inference to provide per-modality attribution scores, enabling the ablation experiments to quantify the marginal contribution of each block to predictive accuracy.

== Synthesis: Gap Table and Research Hypotheses

No prior published study has jointly fused financial fundamentals, FinBERT textual embeddings, and supply-chain graph topology, directed this multimodal architecture at binary CAR direction classification as a measure of post-merger synergy, and tested the marginal contribution of each modality through controlled ablation. The gap table above maps the precise points at which prior work reaches its limits. The three formal hypotheses below emerge directly from those limits.

=== Formal Research Hypotheses

*H1 — The Topological Alpha Hypothesis:* The inclusion of second-order neighbour embeddings via GraphSAGE (Block C) will yield a statistically significant increase in AUC-ROC relative to the financial-only baseline (Block A), under five-fold stratified cross-validation. This gain will be disproportionately concentrated within supply-chain-dependent manufacturing sectors (SIC 20–49) compared to asset-light service sectors (SIC 60–79), reflecting the hypothesis that graph embeddings recover signal proportional to industrial ecosystem structural density @cohen2008 @ahern2014.

*H2 — The Semantic Divergence Hypothesis:* The predictive relationship between textual similarity and synergy direction is conditional on document section. $ "CAR"_i = beta_0 + beta_1 dot "sim"_("MDA",i) + beta_2 dot "sim"_("RF",i) + epsilon_i $
and directly tests the monotonic sentiment utility assumption embedded in standard NLP pipelines @loughran2011 @hajek2024.

*H3 — The Topological Arbitrage Hypothesis:* Acquirer nodes with high betweenness centrality in the heterogeneous supply-chain graph will exhibit statistically compressed variance in $|"CAR"|$ outcomes relative to peripheral nodes, as measured by Levene's test across betweenness centrality quantile groups. This tests the Information Transparency Dampening mechanism: structurally prominent acquirers are more continuously monitored by the market, making their deal announcements less surprising and their returns less dispersed @larcker2013 @ahern2014.

#bibliography("works-litreview.bib", style: "ieee")
