// ============================================================
//  revised-methodology.typ
//  Chapter 3: Methodology (Revised and Defended)
//  M&A Synergy Prediction | Hard Joshi | UEL
// ============================================================

#import "@preview/cetz:0.3.4": canvas, draw

= Methodology

== Introduction

This chapter specifies the final methodological design implemented in the study and defends each design choice against the principal econometric and machine learning objections that a reviewer could reasonably raise. The chapter is written from the standpoint of _completed implementation_: the architecture, preprocessing logic, event-study specification, and evaluation pipeline described below correspond to the final codebase used to generate the reported results. The aim is therefore not to present an idealised methodology in the abstract, but to justify the concrete methodological compromises required when modelling a sparse, noisy, multimodal M&A dataset within the constraints of real financial data and a master’s dissertation scale @creswell2014.

The chapter proceeds in six parts. Section 3.2 positions the study philosophically and explains its deductive quantitative design. Section 3.3 documents the data pipeline, including deal sampling, event-study label construction, temporal splitting, and modality-specific feature engineering. Section 3.4 defends the modelling architecture, with particular attention to late fusion, dimensionality reduction, and the role of frozen transfer embeddings. Section 3.5 specifies the statistical testing framework for the three hypotheses. Section 3.6 addresses known limitations head-on, especially the event window, market model simplification, and potential leakage risks. Section 3.7 concludes with ethical and reproducibility considerations.

== Research Philosophy and Design Logic

=== Philosophical Positioning

The study adopts a post-positivist research philosophy @creswell2014. This position is appropriate because the object of interest — post-merger synergy — is assumed to be real, but only indirectly observable through imperfect market-based proxies. In other words, synergy exists as an economic phenomenon, yet its empirical measurement is contaminated by behavioural bias, information asymmetry, and market microstructure noise, all of which prevent perfectly clean observation @martynova2008 @roll1986 @akerlof1970. The methodological consequence is that inference must be probabilistic rather than absolute: the study evaluates whether the proposed multimodal architecture improves discrimination of positive versus negative value creation, not whether it can recover a metaphysically exact synergy quantity.

The epistemological stance is deductive. The hypotheses were derived from the failures identified in Chapter 2: first, that topological information omitted from finance-only models contains predictive alpha; second, that textual similarity has section-dependent rather than monotonic effects; and third, that network position dampens CAR dispersion. The methodological role of Chapter 3 is therefore not exploratory pattern hunting, but the construction of a controlled empirical pipeline capable of subjecting those claims to falsifiable statistical tests @betton2008 @mackinlay1997.

=== Quantitative Empirical Strategy

The study sits within the empirical corporate finance tradition of event studies and predictive modelling. The core label is derived from abnormal announcement-period returns, consistent with the standard finance interpretation that short-window market reactions capture the present value of expected deal synergies under semi-strong market efficiency @fama1991 @mackinlay1997. This event-study tradition is combined with a machine-learning prediction framework because the problem is not purely causal identification; it is high-dimensional discrimination under structural information fragmentation. Classical econometrics defines the target. Machine learning provides the functional approximation required to map heterogeneous features into that target.

This hybrid strategy is necessary because each individual modality is insufficient on its own. Financial ratios encode balance-sheet capacity but not strategic alignment; textual disclosures encode intent but not dependency structure; graphs encode topology but not explicit valuation quality. The study therefore operationalises synergy prediction as a multimodal inference problem, not as a single-model extension of traditional takeover regressions @palepu1986 @loughran2011 @cohen2008 @baltrusaitis2019.

== Data, Sampling, and Target Construction

=== Data Sources

The implementation integrates four data environments. First, the deal universe and core transaction variables are sourced from LSEG Workspace / Refinitiv M&A data. Second, daily stock returns for acquirers and market benchmarks are obtained from CRSP-style market return series available through institutional financial data access, enabling event-study construction in the MacKinlay tradition @mackinlay1997 @brown1985. Third, inter-firm relationship data are assembled from Bloomberg SPLC and related firm-link sources to construct the supply-chain and competition graph. Fourth, annual 10-K filings are retrieved from SEC EDGAR and parsed section-wise to obtain textual disclosures for the acquirer and target firms.

These sources are selected because they map directly onto the three theoretical signal domains established in the literature review: financial capacity, semantic disclosure, and structural topology. The methodology is therefore data-driven only in a weak sense; in a stronger sense it is theory-indexed. Each dataset exists because a specific omission in prior literature was diagnosed and then deliberately addressed.

=== Sample Construction

The analytical sample is restricted to completed M&A deals involving publicly listed U.S. acquirers with sufficient market return history, structured financial information, and at least partial textual and graph coverage. The sample period spans 2010–2023, producing a modern post-crisis deal environment while preserving enough temporal depth for train/validation/test segmentation. Deals without adequate pre-announcement return history are removed because event-study estimation would otherwise become unstable. Deals lacking the necessary filing or relationship coverage are excluded only where the corresponding modality is essential to the model under evaluation.

This filtering logic creates an unavoidable trade-off between sample size and measurement quality. A larger sample assembled from looser criteria would increase statistical power but at the cost of weaker label validity and noisier multimodal coverage. The final implemented design prioritises measurement integrity over maximal deal count, because a poorly measured target variable contaminates every downstream model regardless of its sophistication @martynova2008 @betton2008.

=== Event Study Label Design

The binary label is derived from Cumulative Abnormal Return (CAR) around the deal announcement. The abnormal return for firm $i$ on day $t$ is defined using a single-factor market model:

$AR_(i,t) = R_(i,t) - (hat(alpha)_i + hat(beta)_i R_(m,t))$

and cumulative abnormal return is:

$CAR_i = sum_(t=t_1)^(t_2) AR_(i,t)$

The primary implementation uses the event window $[-5,+5]$, with the label defined as:

$y_i = cases(1 & if\ CAR_i > 0, 0 & otherwise)$

This formulation intentionally predicts _direction_ rather than _magnitude_. That choice follows a core argument established in Chapter 2: the exact magnitude of announcement-period value creation is heavily contaminated by market timing, bidder overvaluation, competing bids, and general return volatility, whereas the sign of the market reaction remains a more stable proxy for whether the deal was interpreted as value-creating or value-destructive @shleifer2003 @betton2008 @fama1991.

=== Defence of the Event Window

*Examiner callout — event window choice:* The decision to use $[-5,+5]$ rather than the narrower $[-1,+1]$ window deserves explicit defence. A narrow three-day window is the classical response to contamination risk, and many event studies prefer it precisely because it isolates immediate surprise more cleanly @brown1985 @mackinlay1997. That objection is valid. However, in M&A contexts a purely narrow window also risks missing economically meaningful information leakage and slower institutional incorporation. Takeover information frequently diffuses before the formal announcement through rumours, strategic press coverage, analyst anticipation, and abnormal pre-bid trading, while part of the market reaction may complete only after the first trading day due to uncertainty over integration plausibility and payment structure @betton2008.

For this reason, the implemented study uses $[-5,+5]$ as the _primary_ operational window but does not present it as uncontestable. Instead, the chapter explicitly recognises the contamination risk and frames the wider window as a trade-off between leakage capture and noise exposure. To neutralise the reviewer’s strongest objection, robustness checks are reported using a tighter $[-1,+1]$ window. The methodological claim is therefore not that $[-5,+5]$ is universally superior, but that the reported signal is not an artefact of an arbitrarily generous horizon if the core directional findings remain consistent under the tighter specification @mackinlay1997 @brown1985.

=== Defence of the Market Model: Why Single-Factor OLS Is Sufficient at This Scale

*Examiner callout — Fama-French omission:* The expected return model is also a potential point of attack. A single-factor market model is simpler than a Fama-French three-factor (MKT, SMB, HML) @fama1991 or five-factor (adding RMW, CMA) alternative, and a reviewer may object that a broad market index is an imperfect beta proxy for smaller or mid-cap acquirers, for whom size and value premia produce systematic return components that the single market factor cannot absorb @fama1991. That criticism is theoretically legitimate. Nonetheless, the market model remains the implemented specification for three reasons.

First, the market model remains the canonical event-study baseline because of its transparency, computational tractability, and well-understood statistical behaviour under short windows @mackinlay1997 @brown1985. Second, the event windows used here are sufficiently short that the incremental benefit of adding size and value factors is often modest relative to the dominant announcement shock itself, especially when the downstream task is binary CAR direction rather than precision estimation of abnormal return magnitude. Third, at the implemented dissertation scale, factor-model expansion would materially increase engineering complexity and data dependency without proportionate conceptual gain for the central research question, which concerns multimodal predictive uplift rather than asset-pricing model comparison.

This does not mean the single-factor model is perfect. The methodology therefore acknowledges the omission of Fama-French factors as a bounded econometric simplification and treats it as a limitation rather than hiding it. The defence is pragmatic, not dogmatic: the market model is sufficient for a short-window classification target, but not claimed to be the last word in expected return modelling @fama1991 @mackinlay1997.

== Preprocessing and Leakage Control

=== Temporal Splitting Before Model Fitting

All data are partitioned chronologically into training, validation, and held-out test periods. This temporal split is non-negotiable in financial prediction because random shuffling allows future-distribution information to contaminate past estimation, producing a model that is valid only retrospectively. The final implementation therefore preserves historical order in every evaluation stage: models are trained on earlier deals, tuned on later-but-still-past deals, and finally evaluated on the most recent unseen period.

This temporal design mirrors the real deployment problem. At the moment a new deal is announced, the model cannot access future distributions, future median values, or future covariance structure. Any evaluation protocol that allows such information into training is not merely optimistic; it is economically invalid.

=== Imputation and Standardisation Inside the Cross-Validation Loop

One of the most serious machine-learning risks in tabular finance is silent preprocessing leakage. If missing values are imputed using a median computed over the full training dataset _before_ cross-validation, or if scaling parameters are estimated once and then reused across folds, then each fold’s holdout segment has influenced the transformation applied to its own predictor space. That is a subtle but real leakage channel.

*Examiner callout — data leakage patch:* The final implemented pipeline addresses this explicitly: imputation and standardisation are executed strictly _within_ the cross-validation loop — the median is recomputed on the four inner training folds and applied to the single held-out fold in every iteration, never the reverse. For every fold, the median is recomputed using only the fold’s inner training subset, and the resulting imputer is applied to that fold’s holdout portion. The same rule applies to any scaling transformation. No fold-level holdout observation contributes to its own preprocessing statistics. Concretely: in a 5-fold setup, fold $k$'s holdout is excluded when computing the imputation median and scaling parameters for that fold's training pass — the same principle that prevents a time-series model from peeking at future returns. This is academically important because it prevents the model from benefiting from distributional information that would not be available at inference time.

The same anti-leakage principle also governs temporal validation and test evaluation. Any transformation object — imputer, scaler, PCA basis, feature selector, or threshold calibration rule — is fit on the relevant training data only and then applied forward. This point must be stated explicitly in the dissertation because leakage in preprocessing pipelines is one of the most common reasons financial ML papers report inflated performance.

== Modality-Specific Feature Construction

=== Financial Features

The financial block encodes acquirer quality, target quality, deal structure, and market context using ratios and transaction-level indicators available prior to announcement. These include measures of leverage, liquidity, profitability, valuation, acquisition premium, payment method, and relative deal size. The design rationale is conventional but necessary: these variables capture the bidder’s absorptive capacity and the baseline economic plausibility of the transaction, making them the natural reference block for any ablation comparison @palepu1986 @barnes1990 @zhang2024.

The thesis does not claim novelty in financial feature engineering alone. On the contrary, this block exists partly as a deliberately strong baseline. If the graph and text modalities do not outperform a competent finance-only model, the multimodal thesis collapses. This is why the financial block is designed to be comprehensive enough to constitute a serious benchmark rather than a strawman.

=== Textual Features and the PCA Critique

The textual block is built from section-specific FinBERT embeddings extracted from the acquirer’s and target’s 10-K disclosures, focusing particularly on MD&A and Risk Factors sections. This section split is essential because the methodology rejects the standard NLP assumption that all semantic similarity is uniformly beneficial. Strategic alignment and shared risk exposure are not the same signal; indeed, H2 predicts that their coefficients point in opposite directions. The text pipeline therefore preserves section semantics before fusion rather than collapsing the filing into a single undifferentiated sentiment score @araci2019 @loughran2011 @hajek2024.

The most vulnerable design choice in this block is the use of PCA on FinBERT embeddings. The critique is obvious: FinBERT learns contextual, non-linear semantic manifolds, while PCA is a linear variance-maximising projection. Compressing a 768-dimensional contextual embedding with PCA risks flattening the very geometry that makes transformers powerful. This objection is intellectually serious and must be confronted directly.

The defence is not that PCA is semantically innocent. It is that PCA functions here as a deliberate _regularisation instrument_. Financial text is extremely noisy, highly repetitive, and full of boilerplate disclosure language; in a small-sample dissertation setting, retaining the full embedding dimensionality would dramatically raise the risk that the downstream classifier overfits to sparse linguistic artefacts rather than stable semantic structure. PCA was therefore selected not because it preserves every nuance of the transformer manifold, but because it imposes a deterministic, variance-ranked compression that strips out low-energy embedding directions most likely to encode idiosyncratic or spurious textual noise. In other words, PCA is treated as a harsh but disciplined bottleneck, sacrificing some representational richness in exchange for generalisation stability under severe sample constraints @baltrusaitis2019 @chen2016.

This is a methodological compromise, but a defensible one. The key claim is modest: not that PCA is optimal for transformer embeddings in general, but that under low-$N$, high-dimensional financial text, deterministic linear compression provides a more credible downstream learning substrate than unconstrained full-dimensional transfer vectors. The same logic that rejects end-to-end fine-tuning because of overfitting risk also justifies PCA as a regularised bridge between contextual semantics and tabular fusion.

=== Graph Construction and HeteroGraphSAGE

The graph block models firms as nodes embedded in a heterogeneous industrial network containing supply, customer, and competition relations. This design follows the theoretical claim that value creation in M&A depends not only on firm-level balance sheets but also on ecosystem structure, contagion pathways, and positional advantage within a dependency network @cohen2008 @fee2004 @ahern2014. Homogeneous graph treatment would collapse economically distinct relations into a single edge type and thereby discard the very semantics the topology is meant to recover.

GraphSAGE is selected over transductive alternatives because the M&A problem contains an inductive cold-start structure: many potential targets have sparse histories or appear only partially within the observed graph. GraphSAGE learns neighbourhood aggregation functions rather than memorising node identities, making it more suitable for unseen or weakly observed firms @hamilton2017. The heterogeneous extension is justified because `supplier_of`, `customer_of`, and `competes_with` relationships do not represent the same economic mechanism; their predictive contribution should therefore be learned through separate relational channels rather than pooled indiscriminately @wang2019han @shi2017.

The graph embedding is not claimed to be a direct estimator of synergy. Rather, it is a learned representation of structural context. That distinction matters for the fusion logic explained below.

== Model Architecture and the Late-Fusion Defence

=== Why Joint End-to-End Training Was Rejected

A natural reviewer question is why the study does not jointly fine-tune text, graph, and classification layers end-to-end using CAR as the final loss. In theory, joint optimisation is attractive because it allows representation learning to align directly with the target objective. In practice, this was rejected because the sample size and signal quality of M&A data make such training unstable. Event-study labels are inherently noisy, the class signal is weak, and the number of complete multimodal observations is in the low-thousands rather than the millions typically required for robust end-to-end multimodal deep learning @baltrusaitis2019.

The implemented system therefore adopts decoupled late fusion: financial, textual, and structural representations are computed first, then fused by a downstream classifier. This is not a retreat from theoretical ambition; it is a methodological acknowledgement that forcing all representation learning stages to optimise directly on a noisy CAR label would likely produce memorisation, not discovery.

=== The Overfitting Paradox and Inductive Transfer Framing

The most serious conceptual criticism of late fusion is that frozen FinBERT and GraphSAGE embeddings were not originally trained to predict M&A synergy. FinBERT is pretrained on financial language modelling, not CAR. GraphSAGE learns general relational structure, not merger value creation. A sceptical reviewer can therefore argue that the embeddings do not intrinsically “know” anything about synergy.

That criticism is correct in a narrow sense, and the chapter should concede it. The methodological defence is that the architecture is intentionally framed as an _inductive transfer learning problem_. The upstream models are not expected to encode synergy directly; they encode reusable semantic and topological priors. The downstream learner — in the final implementation, XGBoost and associated comparative fusion models — performs the task-specific mapping from those general representations to the highly specific financial outcome of interest @chen2016 @araci2019 @hamilton2017.

This transfer-learning framing is stronger than pretending the embeddings are already synergy-aware. It states, more precisely, that the research hypothesis is about whether generic but economically relevant latent structure can be _translated_ into synergy-predictive alpha when combined with a competent supervised learner. The claim is therefore not “FinBERT predicts synergy” or “GraphSAGE predicts synergy,” but rather “their latent representations contain information that a supervised fusion layer can convert into synergy discrimination.”

=== Why SHAP Becomes Methodologically Central

Once the architecture is framed this way, interpretability is no longer optional. If frozen transfer embeddings are being mapped to CAR direction by a downstream classifier, then post-hoc explanation is the only mechanism by which the thesis can demonstrate that the mapping is economically meaningful rather than accidental. This is why SHAP is elevated from a dashboard convenience to a methodological necessity.

SHAP values provide a game-theoretic decomposition of each feature’s marginal contribution to predictions, allowing the study to test whether the frozen semantic and graph representations actually contribute explanatory mass beyond finance-only variables @lundberg2017. This matters not only for model interpretability but also for epistemic defence: if the transfer features never appear among the dominant SHAP contributors, then the multimodal architecture has failed substantively even if headline AUC improves marginally. Conversely, if graph- and text-derived components consistently appear among the top contributors, they provide the mathematical evidence that the transferred embeddings contain synergy-relevant signal. In this architecture, SHAP is therefore the evidentiary bridge between predictive performance and economic credibility.

== Evaluation Design and Hypothesis Testing

=== Model Ladder and Ablation Logic

The methodology evaluates models in a structured ladder rather than a single headline comparison. The financial-only baseline establishes the tabular ceiling. The finance+text and finance+graph variants test whether each omitted modality independently adds predictive value. The full multimodal model then tests whether the three domains carry complementary rather than redundant variance. This ablation logic is critical because it prevents the thesis from hiding behind a monolithic final model whose gains cannot be attributed.

The primary evaluation metric is AUC-ROC, with Accuracy and F1 reported as secondary diagnostics. AUC-ROC is preferred because the decision threshold is arbitrary and because the economic cost of false positives and false negatives is asymmetric. A threshold-invariant metric is therefore more appropriate than raw accuracy for comparing probabilistic ranking performance across model families @betton2008 @zhang2024.

=== Hypothesis Tests

_H1: Topological Alpha._ The incremental value of graph features is tested by comparing graph-augmented models against finance-only baselines under the same fold structure. The hypothesis is that second-order structural information produces statistically significant AUC improvement, particularly in supply-chain-dense sectors.

_H2: Semantic Divergence._ The section-conditioned effect of textual similarity is tested using regression of raw CAR on MD&A similarity and Risk Factor similarity. The sign asymmetry is central: positive alignment in strategic narrative is hypothesised to differ from concentrated exposure in risk discourse.

_H3: Topological Arbitrage._ The dampening hypothesis is evaluated by testing whether firms with high betweenness centrality exhibit compressed variance in absolute CAR relative to peripheral firms. This is treated as a structural market-position effect rather than a directional return effect.

The important methodological point is that these tests are not afterthoughts added to justify a black-box model. They are the inferential scaffolding that links the architecture back to the theory of omitted signal domains developed in the literature review.

== Limitations Faced and Defended

=== Event-Window Contamination Risk

The first limitation is the possibility that the $[-5,+5]$ CAR window captures confounding non-deal news. This risk is real and cannot be eliminated entirely in any short-horizon event study @mackinlay1997 @brown1985. The defence is twofold: the wider window is theoretically motivated by leakage and gradual incorporation, and robustness analysis with $[-1,+1]$ is used to demonstrate that the central directional results are not window-specific.

=== Simplified Expected Return Model

The second limitation is the use of a single-factor market model instead of a multi-factor asset-pricing specification. This may under-adjust abnormal returns for style exposures, especially in smaller acquirers. The study acknowledges this directly and treats it as a bounded modelling simplification rather than an overlooked flaw @fama1991. Given the short event horizon and binary target design, the simplification is considered acceptable, but not invisible.

=== Information Loss from PCA Compression

The third limitation is that PCA inevitably discards part of the transformer representation. This is conceded explicitly. The defence is that under severe small-sample multimodal conditions, the relevant optimisation problem is not representational completeness but out-of-sample generalisation. PCA is therefore defended as deliberate dimensional austerity rather than accidental simplification.

=== Transfer Misalignment in Frozen Embeddings

The fourth limitation is that the transferred text and graph embeddings are not supervised directly on CAR. This is acknowledged and reframed through inductive transfer learning. Their validity is not assumed; it is empirically tested through ablation gains and SHAP contribution analysis.

== Ethics, Reproducibility, and Finality of Implementation

The study uses corporate, market, and relational data obtained through institutional and public sources. No personal or sensitive human-subject data are involved. The principal ethical obligations therefore concern licensing compliance, reproducibility, and truthful reporting of model limitations. Proprietary datasets are not redistributed in raw form. Instead, only derived results, aggregate statistics, and model outputs are reported.

Reproducibility is strengthened by the fact that the implementation is complete and frozen. The GitHub repository contains the final code used for preprocessing, modelling, and evaluation, and the methodology chapter has been revised to describe that implemented pipeline rather than a hypothetical superior version. This is a strength, not a weakness: a dissertation becomes more defensible when the text matches the code exactly. The purpose of this revised methodology is therefore to make every compromise visible, justified, and theoretically anchored, so that the final thesis cannot be dismissed as a loosely specified engineering prototype.
