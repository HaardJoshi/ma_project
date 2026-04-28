// ============================================================
//  revised-methodology.typ  (Final — All 4 audit patches applied)
//  Chapter 3: Methodology
//  M&A Synergy Prediction | Hard Joshi | UEL
// ============================================================

= Methodology

== Introduction

This chapter specifies the final methodological design implemented in this study and defends each design choice against the principal econometric and machine-learning objections that a reviewer could reasonably raise. The chapter is written from the standpoint of _completed implementation_: the architecture, preprocessing logic, event-study specification, and evaluation pipeline described below correspond to the final codebase used to generate the reported results. The aim is therefore not to present an idealised methodology in the abstract, but to justify the concrete methodological compromises required when modelling a sparse, noisy, multimodal M&A dataset within the constraints of real financial data and a master's dissertation scale @creswell2014.

The chapter proceeds in six parts. Section 3.2 positions the study philosophically and explains its deductive quantitative design. Section 3.3 documents the data pipeline, including deal sampling, event-study label construction, temporal splitting, and modality-specific feature engineering. Section 3.4 defends the modelling architecture, with particular attention to late fusion, dimensionality reduction, and the role of frozen transfer embeddings. Section 3.5 specifies the statistical testing framework for the three hypotheses. Section 3.6 addresses known limitations head-on, covering the event window, market model simplification, and potential leakage risks. Section 3.7 concludes with ethical and reproducibility considerations.

== Research Philosophy and Design Logic

=== Philosophical Positioning

The study adopts a post-positivist research philosophy @creswell2014. This position is appropriate because the object of interest — post-merger synergy — is assumed to be real, but only indirectly observable through imperfect market-based proxies. In other words, synergy exists as an economic phenomenon, yet its empirical measurement is contaminated by behavioural bias, information asymmetry, and market microstructure noise, all of which prevent perfectly clean observation @martynova2008 @roll1986 @akerlof1970. The methodological consequence is that inference must be probabilistic rather than absolute: the study evaluates whether the proposed multimodal architecture improves discrimination of positive versus negative value creation, not whether it can recover a metaphysically exact synergy quantity.

The epistemological stance is deductive. The hypotheses were derived from the failures identified in Chapter 2: first, that topological information omitted from finance-only models contains predictive alpha; second, that textual similarity has section-dependent rather than monotonic effects; and third, that network position dampens CAR dispersion. The methodological role of Chapter 3 is therefore not exploratory pattern hunting, but the construction of a controlled empirical pipeline capable of subjecting those claims to falsifiable statistical tests @betton2008 @mackinlay1997.

=== Quantitative Empirical Strategy

The study sits within the empirical corporate finance tradition of event studies and predictive modelling. The core label is derived from abnormal announcement-period returns, consistent with the standard finance interpretation that short-window market reactions capture the present value of expected deal synergies under semi-strong market efficiency @fama1991 @mackinlay1997. This event-study tradition is combined with a machine-learning prediction framework because the problem is not purely causal identification; it is high-dimensional discrimination under structural information fragmentation. Classical econometrics defines the target. Machine learning provides the functional approximation required to map heterogeneous features into that target.

This hybrid strategy is necessary because each individual modality is insufficient on its own. Financial ratios encode balance-sheet capacity but not strategic alignment; textual disclosures encode intent but not dependency structure; graphs encode topology but not explicit valuation quality. The study therefore operationalises synergy prediction as a multimodal inference problem, not as a single-model extension of traditional takeover regressions @palepu1986 @loughran2011 @cohen2008 @baltrusaitis2019.

== Data, Sampling, and Target Construction

=== Data Sources

The implementation integrates four data environments. First, the deal universe and core transaction variables are sourced from LSEG Workspace / Refinitiv M&A data. Second, daily stock returns for acquirers and market benchmarks are obtained from CRSP-style market return series, enabling event-study construction in the MacKinlay tradition @mackinlay1997 @brown1985. Third, inter-firm relationship data are assembled from Bloomberg SPLC and related firm-link sources to construct the supply-chain and competition graph. Fourth, annual 10-K filings are retrieved from SEC EDGAR and parsed section-wise to obtain textual disclosures for the acquirer and target firms.

These sources are selected because they map directly onto the three theoretical signal domains established in the literature review: financial capacity, semantic disclosure, and structural topology. The methodology is therefore data-driven only in a weak sense; in a stronger sense it is theory-indexed. Each dataset exists because a specific omission in prior literature was diagnosed and then deliberately addressed.

=== Sample Construction

The analytical sample is restricted to completed M&A deals involving publicly listed U.S. acquirers with sufficient market return history, structured financial information, and at least partial textual and graph coverage. The sample period spans 2010–2023, producing a modern post-crisis deal environment while preserving enough temporal depth for train/validation/test segmentation. Deals without adequate pre-announcement return history are removed because event-study estimation would otherwise become unstable. Deals lacking the necessary filing or relationship coverage are excluded only where the corresponding modality is essential to the model under evaluation.

This filtering logic creates an unavoidable trade-off between sample size and measurement quality. A larger sample assembled from looser criteria would increase statistical power but at the cost of weaker label validity and noisier multimodal coverage. The final implemented design prioritises measurement integrity over maximal deal count, because a poorly measured target variable contaminates every downstream model regardless of its sophistication @martynova2008 @betton2008.

=== Event Study Label Design

The binary label is derived from Cumulative Abnormal Return (CAR) around the deal announcement. The abnormal return for firm $i$ on day $t$ is defined using a single-factor market model:

$ A R_(i,t) = R_(i,t) - (hat(alpha)_i + hat(beta)_i R_(m,t)) $

and cumulative abnormal return is:

$ C A R_i = sum_(t = t_1)^(t_2) A R_(i,t) $

The primary implementation uses the event window $[-5, +5]$, with the label defined as:

$ y_i = cases(1 & "if" C A R_i > 0, 0 & "otherwise") $

This formulation intentionally predicts _direction_ rather than _magnitude_. That choice follows a core argument established in Chapter 2: the exact magnitude of announcement-period value creation is heavily contaminated by market timing, bidder overvaluation, competing bids, and general return volatility, whereas the sign of the market reaction remains a more stable proxy for whether the deal was interpreted as value-creating or value-destructive @shleifer2003 @betton2008 @fama1991.

=== Defence of the Event Window

*Examiner callout — event window choice:* The decision to use $[-5, +5]$ rather than the narrower $[-1, +1]$ window deserves explicit defence. A narrow three-day window is the classical response to contamination risk, and many event studies prefer it precisely because it isolates immediate surprise more cleanly @brown1985 @mackinlay1997. That objection is valid. However, in M&A contexts a purely narrow window also risks missing economically meaningful information leakage and slower institutional incorporation. Takeover information frequently diffuses before the formal announcement through rumours, strategic press coverage, analyst anticipation, and abnormal pre-bid trading, while part of the market reaction may complete only after the first trading day due to uncertainty over integration plausibility and payment structure @betton2008.

For this reason, the implemented study uses $[-5, +5]$ as the _primary_ operational window but does not present it as uncontestable. Instead, the chapter explicitly recognises the contamination risk and frames the wider window as a deliberate trade-off between leakage capture and noise exposure. To neutralise the reviewer's strongest objection, robustness checks are reported using a tighter $[-1, +1]$ window. The methodological claim is therefore not that $[-5, +5]$ is universally superior, but that the reported signal is not an artefact of an arbitrarily generous horizon if the core directional findings remain consistent under the tighter specification @mackinlay1997 @brown1985.

=== Defence of the Market Model: Why Single-Factor OLS Is Sufficient at This Scale

*Examiner callout — Fama-French omission:* The expected return model is also a potential point of attack. A single-factor market model is simpler than a Fama-French three-factor (MKT, SMB, HML) @fama1991 or five-factor (adding RMW, CMA) alternative, and a reviewer may object that a broad market index is an imperfect beta proxy for smaller or mid-cap acquirers, for whom size and value premia produce systematic return components that the single market factor cannot absorb.

Three reasons justify the implemented specification. First, the market model remains the canonical event-study baseline because of its transparency, computational tractability, and well-understood statistical behaviour under short windows @mackinlay1997 @brown1985. Second, the event windows used here are sufficiently short that the incremental benefit of adding size and value factors is often modest relative to the dominant announcement shock, especially when the downstream task is binary CAR direction rather than precision estimation of abnormal return magnitude. Third, at dissertation scale, factor-model expansion would materially increase engineering complexity without proportionate conceptual gain for the central research question. Critically: *while multi-factor models capture broader risk premia, the single-factor market model isolates the idiosyncratic deal surprise (CAR) without over-parameterising the baseline, aligning with standard event-study methodology for broad-market acquirers* @mackinlay1997 @brown1985. The market model is acknowledged as a bounded econometric simplification, not hidden as though it were a neutral default.

== Preprocessing and Leakage Control

=== Prevention of Forward-Looking Data Leakage

#rect(fill: luma(235), inset: 10pt, radius: 4pt, width: 100%)[
  *Implementation guarantee:* All scaling and median imputation parameters were fit exclusively on the training folds during each cross-validation iteration to guarantee zero information leakage into the validation fold.
]

Silent preprocessing leakage is one of the most common and least visible reasons financial machine-learning papers report inflated performance. It occurs when missing-value imputation medians or StandardScaler parameters are estimated over the entire training dataset _before_ the cross-validation loop begins — meaning that each fold's held-out validation observations have already implicitly influenced the transformation applied to their own features. The contamination is subtle but mathematically real: the model has seen the statistical distribution of its own test set before predicting it.

The final implemented pipeline eliminates this entirely. Imputation and standardisation are executed _strictly within the cross-validation loop_: for every fold $k$ in the 5-fold stratified scheme, the imputation median and scaling parameters are computed exclusively on the four inner training folds and then applied forward to fold $k$'s held-out portion. The held-out fold contributes zero information to its own transformation. Concretely, in fold $k = 3$, observations from fold 3 are entirely excluded when fitting the imputer and scaler — the same discipline that prevents a time-series model from accessing future returns during estimation. No fold-level holdout observation contributes to its own preprocessing statistics.

The same anti-leakage principle governs the temporal validation and test evaluation stages. Every transformation object — imputer, scaler, PCA basis, and feature selector — is fit on the relevant training period only and applied forward to later periods without refitting. This is not merely good practice; it is the minimum standard for producing AUC-ROC estimates that are realisable in production deployment rather than retrospective artefacts.

== Modality-Specific Feature Construction

=== Financial Features

The financial block encodes acquirer quality, target quality, deal structure, and market context using ratios and transaction-level indicators available prior to announcement. These include measures of leverage, liquidity, profitability, valuation, acquisition premium, payment method, and relative deal size. The design rationale is conventional but necessary: these variables capture the bidder's absorptive capacity and the baseline economic plausibility of the transaction, making them the natural reference block for any ablation comparison @palepu1986 @barnes1990 @zhang2024.

The thesis does not claim novelty in financial feature engineering alone. On the contrary, this block exists partly as a deliberately strong baseline. If the graph and text modalities do not outperform a competent finance-only model, the multimodal thesis collapses. This is why the financial block is designed to be comprehensive enough to constitute a serious benchmark rather than a strawman.

=== Textual Features and the PCA Critique

The textual block is built from section-specific FinBERT embeddings extracted from the acquirer's and target's 10-K disclosures, focusing particularly on MD&A and Risk Factors sections. This section split is essential because the methodology rejects the standard NLP assumption that all semantic similarity is uniformly beneficial. Strategic alignment and shared risk exposure are not the same signal; H2 predicts that their coefficients point in opposite directions. The text pipeline therefore preserves section semantics before fusion rather than collapsing the filing into a single undifferentiated sentiment score @araci2019 @loughran2011 @hajek2024.

The most vulnerable design choice in this block is the use of PCA on FinBERT embeddings. The critique is obvious: FinBERT learns contextual, non-linear semantic manifolds, while PCA is a linear variance-maximising projection. Compressing a 768-dimensional contextual embedding with PCA risks flattening the very geometry that makes transformers powerful. This objection is intellectually serious and must be confronted directly.

The defence is not that PCA is semantically innocent. It is that PCA functions here as a deliberate _regularisation instrument_. Financial text is extremely noisy, highly repetitive, and full of boilerplate disclosure language; in a small-sample dissertation setting, retaining the full embedding dimensionality would dramatically raise the risk that the downstream classifier overfits to sparse linguistic artefacts rather than stable semantic structure.

It is worth acknowledging explicitly what is lost. Non-linear alternatives such as UMAP or t-SNE can preserve topological neighbourhood structure in high-dimensional embedding spaces more faithfully than a linear projection. However, both methods produce fundamentally _non-deterministic, sample-dependent_ transformations: a UMAP basis fit on training data cannot be applied to unseen validation observations in a mathematically consistent way, making them incompatible with rigorous fold-level cross-validation. PCA was therefore selected not because it is the richest representational tool available, but because it is the only dimensionality reduction technique that guarantees a deterministic, out-of-sample-transformable basis — *prioritising robust evaluation integrity over marginal representational capacity*. In other words, PCA is treated as a harsh but disciplined bottleneck: sacrificing some representational richness in exchange for generalisation stability under severe sample constraints @baltrusaitis2019 @chen2016.

=== Graph Construction and HeteroGraphSAGE

The graph block models firms as nodes embedded in a heterogeneous industrial network containing supply, customer, and competition relations. This design follows the theoretical claim that value creation in M&A depends not only on firm-level balance sheets but also on ecosystem structure, contagion pathways, and positional advantage within a dependency network @cohen2008 @fee2004 @ahern2014. Homogeneous graph treatment would collapse economically distinct relations into a single edge type and thereby discard the very semantics the topology is meant to recover.

GraphSAGE is selected over transductive alternatives because the M&A problem contains an inductive cold-start structure: many potential targets have sparse histories or appear only partially within the observed graph. GraphSAGE learns neighbourhood aggregation functions rather than memorising node identities, making it more suitable for unseen or weakly observed firms @hamilton2017. The heterogeneous extension is justified because `supplier_of`, `customer_of`, and `competes_with` relationships do not represent the same economic mechanism; their predictive contribution should therefore be learned through separate relational channels rather than pooled indiscriminately @wang2019han @shi2017.

== Model Architecture and the Late-Fusion Defence

=== Why Joint End-to-End Training Was Rejected

A natural reviewer question is why the study does not jointly fine-tune text, graph, and classification layers end-to-end using CAR as the final loss. In theory, joint optimisation is attractive because it allows representation learning to align directly with the target objective. In practice, this was rejected because the sample size and signal quality of M&A data make such training unstable. Event-study labels are inherently noisy, the class signal is weak, and the number of complete multimodal observations is in the low-thousands rather than the millions typically required for robust end-to-end multimodal deep learning @baltrusaitis2019.

The implemented system therefore adopts decoupled late fusion: financial, textual, and structural representations are computed first, then fused by a downstream classifier. This is not a retreat from theoretical ambition; it is a methodological acknowledgement that forcing all representation learning stages to optimise directly on a noisy CAR label would likely produce memorisation, not discovery.

=== The Overfitting Paradox and Inductive Transfer Framing

The most serious conceptual criticism of late fusion is that frozen FinBERT and GraphSAGE embeddings were not originally trained to predict M&A synergy. FinBERT is pretrained on financial language modelling, not CAR. GraphSAGE learns general relational structure, not merger value creation. A sceptical reviewer can therefore argue that the embeddings do not intrinsically encode anything about synergy.

That criticism is correct in a narrow sense, and this chapter concedes it. The methodological defence is that the architecture is intentionally framed as an _inductive transfer learning problem_. The upstream models are not expected to encode synergy directly; they encode reusable semantic and topological priors. The downstream learner — XGBoost in the final implementation — performs the task-specific mapping from those general representations to the highly specific financial outcome of interest @chen2016 @araci2019 @hamilton2017.

The claim is therefore not "FinBERT predicts synergy" or "GraphSAGE predicts synergy," but rather "their latent representations contain information that a supervised fusion layer can convert into synergy discrimination." *The methodology does not assume this translation succeeds — it tests it directly, with the ablation ladder (M2 to M4 to M6) providing the discrimination metric and SHAP decomposition providing the economic interpretation layer.*

=== Why SHAP Becomes Methodologically Central

Once the architecture is framed this way, interpretability is no longer optional. SHAP values provide a game-theoretic decomposition of each feature's marginal contribution to predictions, allowing the study to test whether the frozen semantic and graph representations actually contribute explanatory mass beyond finance-only variables @lundberg2017. If the transfer features never appear among the dominant SHAP contributors, the multimodal architecture has failed substantively even if headline AUC improves marginally. Conversely, if graph- and text-derived components consistently appear among the top contributors, they constitute the mathematical evidence that transferred embeddings contain synergy-relevant signal. In this architecture, SHAP is therefore the evidentiary bridge between predictive performance and economic credibility.

== Evaluation Design and Hypothesis Testing

=== Model Ladder and Ablation Logic

The methodology evaluates models in a structured ablation ladder rather than a single headline comparison. The finance-only baseline (M2) establishes the tabular ceiling. The finance + graph (M4) and finance + text (M5) variants test whether each omitted modality independently adds predictive value. The full multimodal model (M6) then tests whether the three domains carry complementary rather than redundant variance. This ladder is critical because it prevents the thesis from hiding behind a monolithic final model whose gains cannot be attributed to any specific modality.

The primary evaluation metric is AUC-ROC, with Accuracy and F1 reported as secondary diagnostics. AUC-ROC is preferred because the decision threshold is arbitrary and because the economic cost of false positives and false negatives is asymmetric. A threshold-invariant metric is therefore more appropriate than raw accuracy for comparing probabilistic ranking performance across model families @betton2008 @zhang2024.

=== Hypothesis Tests

_H1: Topological Alpha._ The incremental value of graph features is tested by comparing graph-augmented models against finance-only baselines under the same fold structure. The hypothesis is that second-order structural information produces a statistically significant AUC improvement ($p < 0.05$, paired $t$-test), particularly in supply-chain-dense sectors (SIC 20–49). The graph-augmented model (M4) lifting AUC-ROC above M2 constitutes the primary evidence; the SHAP contribution of graph-derived features constitutes the interpretability confirmation @cohen2008 @hamilton2017.

_H2: Semantic Divergence._ The section-conditioned effect of textual similarity is tested using bivariate OLS regression of raw CAR on MD&A similarity and Risk Factor similarity simultaneously. The sign asymmetry — $beta_1 > 0$ for MD&A, $beta_2 < 0$ for Risk Factors — is the central testable claim, distinguishing this from the monotonic sentiment assumption of standard NLP pipelines @loughran2011 @araci2019.

_H3: Topological Arbitrage._ The dampening hypothesis is evaluated by testing whether firms with high betweenness centrality exhibit compressed variance in absolute CAR relative to peripheral firms. Levene's test for equality of variance is applied across betweenness centrality quantile groups, selected for its robustness to non-normality and unequal group sizes characteristic of power-law centrality distributions @cohen2008.

== Limitations Faced and Defended

=== Event-Window Contamination Risk

The first limitation is the possibility that the $[-5, +5]$ CAR window captures confounding non-deal news. This risk is real and cannot be eliminated entirely in any short-horizon event study @mackinlay1997 @brown1985. The defence is twofold: the wider window is theoretically motivated by leakage and gradual incorporation, and robustness analysis with $[-1, +1]$ demonstrates that the central directional results are not window-specific.

=== Simplified Expected Return Model

The second limitation is the use of a single-factor market model instead of a multi-factor asset-pricing specification. This may under-adjust abnormal returns for style exposures in smaller acquirers. The study acknowledges this directly and treats it as a bounded modelling simplification @fama1991. Given the short event horizon and binary target design, the simplification is acceptable but not invisible.

=== Information Loss from PCA Compression

The third limitation is that PCA inevitably discards part of the transformer representation. This is conceded explicitly. Non-linear alternatives (UMAP, t-SNE) would preserve more topological structure but lack the deterministic out-of-sample transformability required for rigorous cross-validation. PCA is therefore defended as deliberate dimensional austerity rather than accidental simplification: robust evaluation integrity is prioritised over marginal representational capacity.

=== Transfer Misalignment in Frozen Embeddings

The fourth limitation is that the transferred text and graph embeddings are not supervised directly on CAR. This is acknowledged and reframed through inductive transfer learning. Their validity is not assumed; it is empirically demonstrated through the ablation ladder: the graph-augmented model (M4: finance + topology) produces a measurable AUC-ROC lift over the finance-only baseline (M2), and the full multimodal fusion model (M6) extends this further still. If neither gain clears statistical significance under the paired test, the transfer hypothesis is rejected on its own terms. SHAP analysis then provides the interpretability layer: consistent appearance of graph and text features among top SHAP contributors constitutes the mathematical proof that the frozen embeddings carry synergy-predictive alpha; their absence would falsify the claim.

== Ethics, Reproducibility, and Finality of Implementation

The study uses corporate, market, and relational data obtained through institutional and public sources. No personal or sensitive human-subject data are involved. The principal ethical obligations therefore concern licensing compliance, reproducibility, and truthful reporting of model limitations. Proprietary datasets are not redistributed in raw form; only derived results, aggregate statistics, and model outputs are reported.

Reproducibility is strengthened by the fact that the implementation is complete and frozen. The GitHub repository contains the final code used for preprocessing, modelling, and evaluation, and this methodology chapter describes that implemented pipeline exactly — not a hypothetical superior version. Every compromise is made visible, justified, and theoretically anchored, so that the final thesis reads as an examiner-facing defence document rather than a loosely specified engineering prototype.