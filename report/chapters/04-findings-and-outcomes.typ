// #import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge
// #import "@preview/cetz:0.2.2" as cetz

// = Chapter 3: Methodology

// == 3.1 Research Design Overview

// This study employs a mixed-methods quantitative research design that integrates event-study analysis with multimodal machine learning to investigate the predictability of post-merger synergy. The central thesis is that mergers and acquisitions (M&A) synergy --- quantified by the market's reaction to deal announcements --- cannot be adequately captured by any single analytical modality. Instead, a comprehensive assessment requires the simultaneous consideration of three distinct information channels: (i) *financial fundamentals*, which encode the quantitative capacity and health of the merging firms; (ii) *textual disclosures*, which capture qualitative strategic intent, risk awareness, and managerial narrative; and (iii) *supply chain topology*, which encodes relational capital, ecosystem positioning, and inter-firm dependencies.

// The methodology proceeds in five sequential stages, each feeding the next:

// + *Data Collection* --- Assembly of deal-level financial data from Bloomberg, textual disclosures from SEC EDGAR, and inter-firm supply chain relationships from Bloomberg SPLC.
// + *Target Variable Construction* --- Computation of Cumulative Abnormal Returns (CAR) via market model event-study methodology to quantify post-announcement synergy.
// + *Feature Engineering* --- Three parallel pipelines extract modality-specific representations: raw financial ratios, FinBERT-derived text embeddings, and HeteroGraphSAGE-derived topological embeddings.
// + *Multimodal Fusion & Classification* --- A decoupled late-fusion architecture concatenates the three feature vectors and feeds them into gradient-boosted tree classifiers (XGBoost) for binary synergy prediction.
// + *Evaluation & Hypothesis Testing* --- Stratified cross-validation, statistical significance testing, and SHAP-based interpretability analysis to test three pre-registered hypotheses.

// #figure(
//   caption: [End-to-End System Architecture --- Decoupled Representation Learning and Gradient Boosting Framework.],
//   diagram(
//     node-stroke: 1pt, node-fill: white, edge-stroke: 1pt, node-corner-radius: 2pt,
    
//     node((0,0), [Bloomberg \ (Financial)], shape: fletcher.shapes.cylinder),
//     node((0,1), [SEC EDGAR \ (Text)], shape: fletcher.shapes.cylinder),
//     node((0,2), [Bloomberg SPLC \ (Graph)], shape: fletcher.shapes.cylinder),

//     node((2,0), [StandardScaler], shape: "rect"),
//     node((2,1), [FinBERT + PCA], shape: "rect"),
//     node((2,2), [HeteroGraphSAGE], shape: "rect"),
    
//     node((3,1), [Freeze 🔒], stroke: (dash: "dashed"), shape: "rect"),
//     node((3,2), [Freeze 🔒], stroke: (dash: "dashed"), shape: "rect"),

//     node((5,1), [$\oplus$ Concat \ (249-dim)], shape: "rect"),
//     node((7,1), [XGBoost \ Classifier], shape: "rect"),
//     node((9,1), [Synergy Direction \ (CAR > 0)], shape: "rect"),

//     edge((0,0), (2,0), "-|>"),
//     edge((0,1), (2,1), "-|>"),
//     edge((0,2), (2,2), "-|>"),
    
//     edge((2,0), (5,1), "-|>"),
//     edge((2,1), (3,1), "-|>"),
//     edge((3,1), (5,1), "-|>"),
//     edge((2,2), (3,2), "-|>"),
//     edge((3,2), (5,1), "-|>"),
    
//     edge((5,1), (7,1), "-|>"),
//     edge((7,1), (9,1), "-|>")
//   )
// ) <fig:architecture>

// This design is motivated by the empirical observation that no single feature set yields meaningful predictive performance in isolation. Furthermore, deep learning architectures (MLPs) suffer from catastrophic overfitting on tabular financial datasets with high noise-to-signal ratios ($N=2,864$). Therefore, the architecture was intentionally decoupled into a *Representation Learning pipeline* (using HeteroGraphSAGE and FinBERT to learn structural embeddings) and an *Inference Engine* (using XGBoost). This hybrid approach marries the high-dimensional spatial awareness of neural networks with the robust tabular regularization of tree-based models.

// == 3.2 Data Collection & Sample Construction

// The study draws on three primary data sources, capturing distinct information modalities for US-listed public companies involved in completed M&A transactions.

// === 3.2.1 Sample Overlap & Final Dataset Construction

// Not all data sources have complete overlap. The merging procedure produces progressively filtered subsets, resulting in a final multimodal dataset of 4,999 deals. Deals missing text or graph embeddings retain null values in those dimensions; these are handled through median imputation during the model training phase, ensuring no deal is discarded.

// #figure(
//   caption: [Dataset Construction and Feature Enrichment Pipeline],
//   table(
//     columns: (auto, auto, auto, auto),
//     stroke: none,
//     align: left,
//     table.hline(y: 0, stroke: 1pt),
//     [*Processing Stage*], [*Modality Added*], [*Remaining Deals ($N$)*], [*Feature Space ($RR^d$)*],
//     table.hline(y: 1, stroke: 0.5pt),
//     [Raw Bloomberg Extraction], [Financial Base], [4,999], [67],
//     [CAR Computation Filter], [Target Variable], [4,510], [68],
//     [FinBERT Integration], [Textual (MD&A, RF)], [$approx$ 2,250], [196],
//     [HeteroGraphSAGE Integration], [Topological], [2,864], [268],
//     table.hline(y: 5, stroke: 0.5pt),
//     [*Final Imputed Dataset*], [*Multimodal Fusion*], [*4,999*], [*268*],
//     table.hline(y: 6, stroke: 1pt)
//   )
// ) <tab:dataset_pipeline>


// == 3.3 Target Variable: Cumulative Abnormal Returns (CAR)

// The target variable for this study is the *Cumulative Abnormal Return (CAR)* over the event window surrounding each deal's announcement date. CAR quantifies the stock market's surprise reaction, serving as an objective proxy for the market's assessment of deal-level synergy.

// #figure(
//   caption: [Event Study Methodology Timeline mapping the Estimation and Event Windows.],
//   cetz.canvas({
//     import cetz.draw: *
//     line((0,0), (12,0), mark: (end: ">"))
//     content((12.5, 0), [$t$ (days)])
    
//     line((1, 0.2), (1, -0.2)); content((1, -0.5), [$-260$])
//     line((5, 0.2), (5, -0.2)); content((5, -0.5), [$-11$])
//     line((6.5, 0.2), (6.5, -0.2)); content((6.5, -0.5), [$-5$])
//     line((8, 0.2), (8, -0.2)); content((8, -0.5), [$0$])
//     line((9.5, 0.2), (9.5, -0.2)); content((9.5, -0.5), [$+5$])
    
//     content((8, 0.5), [*Announcement Date*])
    
//     content((3, 1.2), [Estimation Window \ (Market Model: $alpha_i, beta_i$)])
//     line((1, 0.4), (1, 0.8), (5, 0.8), (5, 0.4))
    
//     content((8, -1.5), [Event Window \ (CAR Computation)])
//     line((6.5, -0.4), (6.5, -1.0), (9.5, -1.0), (9.5, -0.4))
//   })
// ) <fig:event_timeline>

// === 3.3.1 Market Model Estimation

// Abnormal returns are estimated using the single-factor market model, which decomposes an acquirer's return into a systematic component and an idiosyncratic component:

// $ R_(i,t) = alpha_i + beta_i R_(m,t) + epsilon_(i,t) $

// The abnormal return for day $t$ is computed as $A R_(i,t) = R_(i,t) - (hat(alpha)_i + hat(beta)_i R_(m,t))$. The CAR is then defined as the sum of these abnormal returns over the $[-5, +5]$ event window:

// $ C A R_i(tau_1, tau_2) = sum_(t=tau_1)^(tau_2) A R_(i,t) $

// Given the severe noise floor inherent in predicting the exact magnitude of $C A R_i$, the problem is formulated as a binary classification task to capture the *directional* synergy outcome:

// $ y_i = cases(1 & "if" C A R_i > 0 quad "(positive synergy)", 0 & "if" C A R_i <= 0 quad "(value destruction)") $


// == 3.4 Feature Engineering Pipeline

// Three parallel feature extraction pipelines transform raw data from each modality into dense numerical representations suitable for machine learning.

// === 3.4.1 NLP Features --- FinBERT Text Embeddings (M2)

// To extract qualitative intent, we utilize FinBERT. The process extracts features from Item 1A (Risk Factors) and Item 7 (MD&A) of pre-merger 10-K SEC filings.

// #figure(
//   caption: [NLP Embedding Pipeline: From 10-K filing to semantic feature vectors.],
//   diagram(
//     node-stroke: 1pt, edge-stroke: 1pt, node-corner-radius: 2pt,
//     node((0,0), [10-K Section \ (MD&A/Risk)], shape: "rect"),
//     node((2,0), [Overlapping Chunks \ (512 tokens)], shape: "rect"),
//     node((4,0), [FinBERT Encoder \ (Transformer Layers)], shape: "rect"),
//     node((6,0), [[CLS] Token \ Extraction], shape: "rect"),
//     node((8,0), [Mean Pooling \ $e_"section" = 1/K sum h_k$], shape: "rect"),
//     node((10,0), [PCA Reduction \ $RR^768 arrow RR^64$], shape: "rect"),
//     node((12,0), [Concat \ $RR^128$], shape: "rect"),
    
//     edge((0,0), (2,0), "-|>"),
//     edge((2,0), (4,0), "-|>"),
//     edge((4,0), (6,0), "-|>"),
//     edge((6,0), (8,0), "-|>"),
//     edge((8,0), (10,0), "-|>"),
//     edge((10,0), (12,0), "-|>")
//   )
// ) <fig:nlp_pipeline>

// === 3.4.2 Graph Features --- HeteroGraphSAGE Embeddings (M3)

// The Bloomberg SPLC data is transformed into a heterogeneous supply chain graph. Node embeddings are learned using a *Heterogeneous GraphSAGE* architecture, extending standard message passing to support multiple edge types (`supplies` and `buys_from`).

// #figure(
//   caption: [Heterogeneous Graph Schema & Message Passing Architecture.],
//   diagram(
//     node-stroke: 1pt, edge-stroke: 1pt, node-corner-radius: 2pt,
    
//     // Schema side
//     node((0,0), [Supplier], shape: "rect"),
//     node((0,1.5), [Acquirer], shape: "circle"),
//     node((0,3), [Customer], shape: "rect"),
    
//     edge((0,0), (0,1.5), "-|>", label: "supplies", label-side: left),
//     edge((0,1.5), (0,3), "-|>", label: "buys_from", label-side: left),
    
//     node((2,1.5), stroke: none, [*Message Passing Math* $arrow$]),
    
//     // Math side
//     node((4,0.5), [$sum "Supplier"_"neighbors"$], shape: "rect"),
//     node((4,2.5), [$sum "Customer"_"neighbors"$], shape: "rect"),
//     node((6,1.5), [$\oplus$ CONCAT], shape: "rect"),
//     node((8,1.5), [$times W^(l)$], shape: "rect"),
//     node((10,1.5), [ReLU ($sigma$)], shape: "rect"),
//     node((12,1.5), [$h_v^(l+1)$], shape: "rect"),
    
//     edge((4,0.5), (6,1.5), "-|>"),
//     edge((4,2.5), (6,1.5), "-|>"),
//     edge((6,1.5), (8,1.5), "-|>"),
//     edge((8,1.5), (10,1.5), "-|>"),
//     edge((10,1.5), (12,1.5), "-|>")
//   )
// ) <fig:graph_schema>

// The network learns representations via a self-supervised link prediction loss function:

// $ cal(L) = -sum_((u,v) in cal(E)^+) log(sigma(h_u^top h_v)) - sum_((u,v') in cal(E)^-) log(1 - sigma(h_u^top h_(v'))) $

// Once trained, the topological embeddings ($RR^64$) are frozen and extracted for downstream fusion.

// == 3.5 Evaluation Protocol & Hypothesis Testing

// The final concatenated vectors are evaluated to test three pre-registered hypotheses. The matrix below outlines the specific empirical strategy deployed to validate the theoretical claims surrounding topological alpha and information dampening.

// #figure(
//   caption: [Summary of Pre-Registered Hypothesis Testing Framework],
//   table(
//     columns: (auto, auto, auto, auto),
//     stroke: none,
//     align: left,
//     table.hline(y: 0, stroke: 1pt),
//     [*Hypothesis*], [*Independent Variable*], [*Dependent Metric*], [*Statistical Test*],
//     table.hline(y: 1, stroke: 0.5pt),
//     [*H1: Topological Alpha*], [Configuration (M1 vs. M3)], [5-Fold AUC-ROC], [Paired $t$-test ($p < 0.05$)],
//     [*H2: Semantic Divergence*], [Cosine Sim. (MD&A, Risk Factors)], [Continuous CAR], [Bivariate OLS Regression],
//     [*H3: Topological Arbitrage*], [Betweenness vs. Clustering Coeff.], [$|CAR|$ Variance], [Levene's Test for Equality],
//     table.hline(y: 4, stroke: 1pt)
//   )
// ) <tab:hypothesis_framework>