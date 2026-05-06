// ============================================================
//  appendix-a.typ
//  Appendix A: Initial and Revised Project Aims and Objectives
//  M&A Synergy Prediction | Hard Joshi | UEL
// ============================================================

= Initial and Revised Project Aims and Objectives <appendix-aims>

This appendix documents the evolution of the project's aims, objectives, and research questions from the initial proposal stage through to the final implementation. The comparison demonstrates how early-stage exploratory scoping was refined into a formally testable, architecturally grounded research programme.

== Initial Project Aims and Objectives (As Proposed)

The original proposal was submitted under the working title _"Predicting M&A Success Using Multimodal Machine Learning"_ and was structured around the following aim and objectives.

*Aim:*
To develop and evaluate a machine learning framework for predicting the likelihood of post-acquisition (M&A) success, moving beyond traditional numbers to incorporate textual data sources.

*Research Gap:*
Existing literature on M&A outcome prediction is fragmented. Different research streams focus on econometric models using historical financial data. These models, while valuable, often do not capture strategic nuances and market sentiment around a deal. A more recent stream applies NLP to textual data but does it in isolation. The distinct research gap lies at the intersection of these concepts. This dissertation addresses that gap by proposing and testing an integrated model, assessing whether a multimodal approach can yield superior predictive accuracy.

*Research Questions:*
+ To what degree can an ML model integrating financial numbers and textual sentiment data improve the accuracy of predicting M&A success compared to traditional models based solely on financial data?
+ How does the sentiment and linguistic specificity of initial M&A announcements correlate with realised post-acquisition performance?
+ How can we achieve a comprehensive understanding of the key features that play an important role in contributing to M&A success?

*Potential Objectives:*
+ Conduct a review of existing literature on M&A valuation, failure factors, and application of ML in financial predictions.
+ Construct a multimodal (textual and numerical) dataset of historical M&A transactions.
+ Design, train, and validate an ML model capable of processing the multimodal dataset.
+ Evaluate the model's predictive performance against a benchmark model.
+ Analyse feature importance for the final model to identify key drivers of the prediction output.

== Evolved Aims and Objectives (Final Implementation)

As the research progressed, the scope expanded to address deeper structural issues identified in the literature, specifically the need to model inter-firm supply chain relationships and section-specific semantics rather than generic sentiment.

*Final Aim:*
To construct, evaluate, and interpret a heterogeneous multimodal machine learning framework (HeteroGraphSAGE + FinBERT + XGBoost) for M&A synergy prediction that transcends traditional tabular limitations by integrating financial fundamentals, section-aware textual semantics, and supply-chain network topology.

*Final Objectives Achieved:*
+ *Literature Synthesis:* Conducted a critical review identifying the structural limitations of isolated analytical streams (tabular, semantic, and topological).
+ *Multimodal Dataset Engineering:* Engineered a unified dataset combining historical M&A financial fundamentals, section-specific 10-K regulatory filings, and inter-firm supply-chain relationship graphs.
+ *Architectural Development:* Designed and trained a late-fusion neural architecture to process this heterogeneous feature space without target leakage.
+ *Benchmark Evaluation:* Evaluated the multimodal model across a dual-pipeline framework (classification and regression), demonstrating a statistically significant AUC-ROC improvement over tabular baselines.
+ *Interpretability:* Utilised SHAP decomposition to interpret the interaction between financial fundamentals and ecosystem topology, validating the economic credibility of the network effects.

== Summary of Key Divergences

#figure(
  table(
    columns: (2fr, 2fr),
    align: (left, left),
    inset: 8pt,
    stroke: 0.5pt,
    table.header(
      [*Initial Proposal*], [*Final Implementation*],
    ),
    [Generic sentiment from news articles], [Section-split FinBERT embeddings from 10-K filings (MDA vs. Risk Factors)],
    [Crunchbase as primary data source], [Yahoo Finance, Bloomberg SPLC, and SEC EDGAR],
    [Binary prediction of "M&A success"], [Binary CAR direction over an 11-day event window],
    [Standard ML pipeline], [Tri-modal late-fusion architecture with HeteroGraphSAGE],
    [Descriptive feature importance], [Formal hypothesis testing (H1, H2, H3) with Bonferroni correction],
  ),
  caption: [Summary of methodological divergences between the initial proposal and final implementation.],
)
