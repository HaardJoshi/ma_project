// ============================================================
//  01-introduction.typ
//  Chapter 1: Introduction
//  M&A Synergy Prediction | Hard Joshi | UEL
// ============================================================



= Introduction <ch-introduction>

== The Problem: A Century of Predictable Failure <sec-intro-problem>

Every year, hundreds of companies announce mergers and acquisitions worth trillions of dollars. Each deal rests on the same fundamental bet: that the combined entity will generate more economic value than the two firms could produce independently. The gap between that bet and its typical outcome is one of the most durable findings in modern corporate finance. Between 70% and 90% of acquisitions fail to generate value for acquirer shareholders, a failure rate that has remained statistically stable across a century of takeover activity and through every wave of increasingly sophisticated due diligence technology @martynova2008. The question this dissertation asks is not *why* individual deals fail - the behavioural and market-timing explanations for that are well-established. The question is structural: what information do all existing predictive frameworks systematically miss, and is that information recoverable?

The answer proposed here is that existing models fail not because of insufficient computational power, but because of a flawed representational assumption. Every major class of quantitative M&A predictor - from logistic regression on financial ratios to gradient-boosted trees and transformer-based NLP pipelines - treats each firm as an isolated data point. This means that no matter how sophisticated the algorithm, it operates in a feature space that is topologically flat: it sees the numbers on the balance sheet but cannot see the economic ecosystem the firm inhabits. A firm's largest customer may be on the verge of bankruptcy. Its target may share fragile single-source suppliers. The combined entity may create a bottleneck in a critical industrial network. None of these facts appear in a financial ratio vector, and none can be recovered by adding more layers to the same architecture.

== Aims and Objectives <sec-intro-aims>

The central aim of this study is to construct, evaluate, and interpret a heterogeneous multimodal machine learning framework for M&A synergy prediction that transcends traditional tabular limitations by integrating financial, textual, and topological data.

The project was guided by the following core objectives:
1. *Literature Synthesis:* To conduct a comprehensive review of existing econometric, NLP, and network-based M&A prediction models, identifying the structural limitations of isolated analytical streams.
2. *Multimodal Dataset Construction:* To engineer a unified dataset combining historical M&A financial fundamentals, section-specific 10-K regulatory filings, and inter-firm supply-chain relationship graphs.
3. *Architectural Development:* To design and train a late-fusion neural architecture capable of processing this heterogeneous feature space.
4. *Benchmark Evaluation:* To rigorously evaluate the multimodal model's predictive performance (AUC-ROC) against a standard financial-only baseline to quantify the incremental value of non-financial signals.
5. *Interpretability and Feature Attribution:* To utilise SHAP (SHapley Additive exPlanations) to deconstruct model predictions, identifying the precise economic drivers of synergy and interpreting the interaction between financial fundamentals and ecosystem topology.

== The Proposed Solution <sec-intro-solution>

This dissertation proposes and empirically tests a multimodal late-fusion architecture designed to break through the tabular ceiling by encoding three complementary information modalities simultaneously:

- *Block A - Financial Fundamentals:* 56 ratio-level features drawn from Yahoo Finance, covering acquirer and target leverage, liquidity, profitability, and deal structure. This is the baseline information available to every prior quantitative model.

- *Block B - Section-Aware Textual Semantics:* FinBERT-encoded embeddings from the Management Discussion & Analysis (MD&A) and Risk Factors sections of pre-announcement 10-K filings, processed separately so that their opposing economic signals are not cancelled by aggregation.

- *Block C - Heterogeneous Graph Topology:* Node embeddings derived from a two-hop HeteroGraphSAGE model trained on Bloomberg SPLC supply-chain relationship data, encoding each firm's structural position within the industrial network it actually inhabits.

The three blocks are fused via late concatenation and evaluated across a dual-pipeline framework: a classifier pipeline that predicts the *direction* of acquirer Cumulative Abnormal Return (CAR) at announcement, and a regression pipeline that attempts to predict CAR *magnitude*. As @ch-findings demonstrates, these two pipelines resolve differently - and that difference is itself a substantive finding. To ensure absolute methodological rigour, the pipeline enforces strict temporal splitting and an 11-day event-window embargo to explicitly prevent target leakage and market microstructure contamination.

== Research Hypotheses <sec-intro-hypotheses>

The study tests three formal hypotheses derived from the four knowledge streams surveyed in @ch-litreview. Each hypothesis corresponds to one of the three modality blocks and is designed to be falsifiable within the available dataset.

*H1 - The Topological Alpha Hypothesis:* The inclusion of heterogeneous supply-chain graph topology will yield a statistically significant improvement in directional deal discrimination (AUC-ROC) over a financial-only baseline, proving that topological data yields orthogonal predictive alpha.

*H2 - The Semantic Divergence Hypothesis:* The relationship between textual similarity and announcement returns is conditional on document section. MD&A similarity (strategic fit) correlates positively with CAR; Risk Factor similarity (shared liability exposure) correlates negatively. These opposing effects are mathematically suppressed when the two sections are pooled.

*H3 - The Topological Arbitrage Hypothesis:* Acquirers with high betweenness centrality in the supply-chain graph exhibit statistically compressed variance in announcement return outcomes relative to peripheral acquirers, as measured by Levene's test across centrality quantile groups.

== What This Study Contributes <sec-intro-contributions>

The academic contribution of this work lies at the intersection of three fields that have not previously been unified in the context of post-merger synergy prediction. Supply chain finance, financial NLP, and heterogeneous graph neural networks have each advanced substantially as independent research programmes. This dissertation is, to the best of current knowledge, the first study to direct their combination at binary CAR direction classification as a proxy for M&A synergy.

The contribution is not only architectural. By reporting the M2 reversal - the empirical finding that adding aggregated FinBERT text *degrades* prediction quality - the study provides a methodological warning to future researchers that NLP is not automatically useful in M&A prediction unless section semantics are preserved. By reporting negative $R^2$ values on the regression pipeline, the study defines the boundary between what multimodal architecture can and cannot achieve in this domain. And by building the Deal Intelligence Terminal, the study operationalises the theoretical framework into a live, reproducible research artefact that renders all empirical claims interactively.

== Structure of the Dissertation <sec-intro-structure>

The dissertation proceeds as follows. @ch-litreview builds a sustained critical argument across four interconnected knowledge streams, demonstrating that each wave of prior scholarship produced an asymptotic ceiling and identifying precisely the structural information each paradigm discarded. @ch-methodology details the full research design, data pipeline, feature construction, and hypothesis-testing protocol. @ch-findings reports the empirical findings across both pipelines, including the negative results that establish the boundary conditions of the architecture. @ch-evaluation evaluates both the research product and the research process honestly, contextualising performance against the literature and identifying what should be done differently in future work. @ch-conclusion synthesises the findings and places the contribution in the context of the broader financial machine learning research agenda.
