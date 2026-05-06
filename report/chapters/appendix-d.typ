// ============================================================
//  appendix-d.typ
//  Appendix D: Deal Intelligence Terminal
//  M&A Synergy Prediction | Hard Joshi | UEL
// ============================================================

= Deal Intelligence Terminal <appendix-terminal>

The empirical results of this dissertation are operationalised through the *Deal Intelligence Terminal*, a full-stack interactive research artefact built in Streamlit and Python. The terminal is designed to bridge the gap between high-dimensional machine learning outputs and actionable financial decision-making. This appendix documents the terminal's interface architecture, the diagnostic interaction flow, and the supplementary research modules used to defend the study's methodological choices.

All screenshots were captured from a live instance of the terminal running against the production dataset described in Chapter 3.

== The Model Evidence Wall

Before performing deal-specific diagnostics, the terminal provides a global *Ablation Wall* to validate the predictive superiority of the multimodal architecture. This interface allows researchers to verify that the inclusion of topological (GraphSAGE) and semantic (FinBERT) features yields a statistically significant lift in AUC-ROC and F1-score over the financial-only baseline, confirming the _Topological Alpha_ and _Semantic Divergence_ hypotheses prior to any deal-level inspection.

#figure(
  image("../terminal_pics/ablation_wall.png", width: 100%),
  caption: [The Ablation Wall interface comparing the M1 (financial-only) baseline against the full M3 (financial + text + graph) architecture across AUC-ROC, F1, precision, and recall.],
)

#pagebreak()

== Deal-Level Diagnostics

The primary interface of the terminal is the *Diagnostic Report*, which provides a "Glass Box" view into a specific M&A transaction. The following screenshots document the diagnostic flow for the Omnicare $arrow$ Theorem Clinical acquisition.

=== The Synergy Forecast

The top-level verdict translates raw model logits into a calibrated probability gauge. Alongside the synergy probability, the terminal displays two *Macro Metrics* for the deal:
- *Network Alpha:* The acquirer's betweenness centrality rank within the global supply-chain graph, which informs the information transparency dampening effect (Hypothesis 3).
- *Semantic Match:* The cosine similarity between acquirer and target MD&A sections, indicating strategic alignment (Hypothesis 2).

#figure(
  image("../terminal_pics/diag_verdict.png", width: 100%),
  caption: [The diagnostic verdict showing a 91% synergy probability calibrated via the M3 ensemble, alongside betweenness centrality rank (30th percentile) and MD&A semantic match.],
)

#pagebreak()

=== Feature Interaction Zones

The middle section of the diagnostic report visualises the internal representation space of the model, separating the _where_ (topology) from the _why_ (semantics).

#figure(
  image("../terminal_pics/diag_zones.png", width: 100%),
  caption: [Zone 1 (Topological Embeddings) rendering the acquirer--target supply-chain subgraph, and Zone 2 (Semantic Radar) plotting the pentagonal linguistic profile comparison.],
)

- *Zone 1 --- Topological Embeddings:* An interactive force-directed graph rendering the acquirer and target nodes within the global supply-chain ecosystem. This allows the user to visually inspect the degree of supplier overlap or customer risk that the GNN is encoding.
- *Zone 2 --- Semantic Radar:* A pentagonal radar chart plotting the acquirer's linguistic profile (MD&A, risk, strategic, sentiment, operational) against the target's. Significant divergence in the Risk axis compared to the Strategic axis provides early warning signals of negative synergy.

#pagebreak()

=== The Glass Box (SHAP Attribution)

The final section of the diagnostic report addresses the "black box" criticism of GNNs and transformers by applying SHAP (SHapley Additive exPlanations) decomposition.

#figure(
  image("../terminal_pics/diag_shap.png", width: 100%),
  caption: [Zone 3: SHAP waterfall decomposition of the synergy probability, with the Algorithmic Translation summary below.],
)

The waterfall plot decomposes the synergy probability into its constituent feature drivers. In this deal, the model identifies *Acquirer Operating Margin* and *Target Net Income* as the primary positive drivers.

The terminal also includes an *Algorithmic Translation* block --- an NLP-driven summary that converts the mathematical SHAP variances into a natural language executive summary. This ensures that the _why_ of the prediction is accessible to non-technical stakeholders without sacrificing the underlying econometric rigour.

#pagebreak()

== Supplementary Research Modules

Beyond the primary diagnostic report, the terminal provides three specialised modules for methodological defence and hypothesis testing.

=== The Methodology Engine

#figure(
  image("../terminal_pics/methodology_engine.png", width: 100%),
  caption: [The Methodology Engine interface, visualising the end-to-end M3 pipeline as an interactive Directed Acyclic Graph (DAG) with stage-specific output statistics.],
)

The Methodology Engine serves as a live DAG of the study's pipeline. It allows inspection of stage-specific output statistics (e.g., number of raw deals loaded versus text-coverage filtered deals) and provides direct links to the underlying Python implementation for each transformation step.

=== The Evaluation Lab

#figure(
  image("../terminal_pics/eval_lab.png", width: 100%),
  caption: [The Evaluation Lab, providing a structured defence of AUC-ROC as the primary performance metric with an interactive ROC curve visualisation.],
)

The Evaluation Lab is designed to address the *Accuracy Paradox* inherent in imbalanced M&A datasets. It demonstrates that while a degenerate classifier could achieve 56% accuracy by predicting "Value Destruction" for all cases, the M3 architecture's AUC of 0.5655 represents genuine predictive signal across threshold-independent decision boundaries.

#pagebreak()

=== The Hypothesis Lab

#figure(
  image("../terminal_pics/hypothesis_lab.png", width: 100%),
  caption: [The Hypothesis Lab, showcasing the sector-wise AUC lift used to validate Hypothesis 1 (Topological Alpha) with $plus.minus 1 sigma$ error bars.],
)

The Hypothesis Lab allows for real-time slicing of model performance across SIC-coded industries. This interface was used to generate the evidence for Hypothesis 1, demonstrating that the GraphSAGE modality provides a disproportionately higher lift in capital-heavy supply-chain sectors ($Delta = +0.059$, $p = 0.0047$) compared to asset-light service industries ($Delta = +0.041$, $p = 0.0329$).

=== Pipeline Architecture

#figure(
  image("../terminal_pics/pipeline_architecture.png", width: 100%),
  caption: [The Pipeline Architecture interface, documenting the internal tensor flow and the 249-dimensional multimodal feature concatenation logic.#footnote[The 249-dimensional pre-projection vector is an intermediate representation; the classifier receives $bold(z)_i in RR^160$ after passing through the modality-specific ProjectionHeads.]],
)

The final module of the terminal provides a high-level overview of the *Multimodal Fusion* head. It visualises the transformation of 56 financial features, 128 textual features (64 MDA + 64 RF), and 65 graph features (64 GraphSAGE embeddings + 1 has-graph flag) into the final 160-dimensional feature vector $bold(z)_i$ consumed by the XGBoost inference engine.
