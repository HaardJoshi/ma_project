#import "@preview/cetz:0.2.2"

#set page(paper: "a4", margin: 1in)
#set text(font: "New Computer Modern", size: 10pt)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

= Chapter 3: Methodology

== 3.1 Research Design Overview

This study employs a mixed-methods quantitative research design that integrates event-study analysis with multimodal machine learning to investigate the predictability of post-merger synergy. The central thesis is that mergers and acquisitions (M&A) synergy --- quantified by the market's reaction to deal announcements --- cannot be adequately captured by any single analytical modality. Instead, a comprehensive assessment requires the simultaneous consideration of three distinct information channels: (i) *financial fundamentals*, which encode the quantitative capacity and health of the merging firms; (ii) *textual disclosures*, which capture qualitative strategic intent, risk awareness, and managerial narrative; and (iii) *supply chain topology*, which encodes relational capital, ecosystem positioning, and inter-firm dependencies.

The methodology proceeds in five sequential stages, each feeding the next:

+ *Data Collection* --- Assembly of deal-level financial data from Bloomberg, textual disclosures from SEC EDGAR, and inter-firm supply chain relationships from Bloomberg SPLC.
+ *Target Variable Construction* --- Computation of Cumulative Abnormal Returns (CAR) via market model event-study methodology to quantify post-announcement synergy.
+ *Feature Engineering* --- Three parallel pipelines extract modality-specific representations: raw financial ratios, FinBERT-derived text embeddings, and HeteroGraphSAGE-derived topological embeddings.
+ *Multimodal Fusion & Classification* --- A decoupled late-fusion architecture concatenates the three feature vectors and feeds them into gradient-boosted tree classifiers (XGBoost) for binary synergy prediction.
+ *Evaluation & Hypothesis Testing* --- Stratified cross-validation, statistical significance testing, and SHAP-based interpretability analysis to test three pre-registered hypotheses.

#figure(
  align(center)[
    #cetz.canvas({
      import cetz.draw
      
      // Data Sources (Column 1)
      draw.rect((0.0, 4.0), (2.5, 5.5), name: "db1")
      draw.content("db1", text(size: 8pt)[*Bloomberg M&A* \ Financial Data])
      draw.rect((0.0, 2.0), (2.5, 3.5), name: "db2")
      draw.content("db2", text(size: 8pt)[*SEC EDGAR* \ Textual Disclosures])
      draw.rect((0.0, 0.0), (2.5, 1.5), name: "db3")
      draw.content("db3", text(size: 8pt)[*Bloomberg SPLC* \ Supply Chain Graph])
      
      // Encoders (Column 2)
      draw.rect((4.5, 4.0), (7.0, 5.5), name: "enc1")
      draw.content("enc1", text(size: 8pt)[*StandardScaler* \ $RR^56$])
      draw.rect((4.5, 2.0), (7.0, 3.5), name: "enc2", stroke: (dash: "dashed"))
      draw.content("enc2", text(size: 8pt)[*FinBERT + PCA* \ $RR^128$ (Frozen)])
      draw.rect((4.5, 0.0), (7.0, 1.5), name: "enc3", stroke: (dash: "dashed"))
      draw.content("enc3", text(size: 8pt)[*HeteroGraphSAGE* \ $RR^64$ (Frozen)])
      
      // Connecting arrows C1 -> C2
      draw.line("db1.right", "enc1.left", mark: (end: ">"))
      draw.line("db2.right", "enc2.left", mark: (end: ">"))
      draw.line("db3.right", "enc3.left", mark: (end: ">"))
      
      // Fusion (Column 3)
      draw.circle((9.0, 2.75), radius: 0.3, name: "fusion")
      draw.content("fusion", [$oplus$])
      draw.line("enc1.right", ("fusion", 135deg), mark: (end: ">"))
      draw.line("enc2.right", "fusion.west", mark: (end: ">"))
      draw.line("enc3.right", ("fusion", 225deg), mark: (end: ">"))
      
      // XGBoost
      draw.rect((10.5, 2.0), (12.5, 3.5), name: "xgb")
      draw.content("xgb", text(size: 8pt)[*XGBoost* \ Classifier])
      draw.line("fusion.east", "xgb.west", mark: (end: ">"), name: "fuse_arrow")
      draw.content("fuse_arrow.mid", text(size: 8pt)[$249$-dim], anchor: "bottom", padding: 0.1)
      
      // Output
      draw.line("xgb.east", (14.0, 2.75), mark: (end: ">"), name: "out")
      draw.content("out.mid", text(size: 8pt)[$hat(y) in {0,1}$], anchor: "bottom", padding: 0.1)
    })
  ],
  caption: [End-to-End System Architecture illustrating decoupled late-fusion.]
)

This design is motivated by the empirical observation that no single feature set yields meaningful predictive performance in isolation --- a finding that itself constitutes a contribution to the M&A literature. The multimodal framework draws on recent advances in heterogeneous information fusion and graph-based relational learning to construct a representation of each deal that is richer than any traditional financial model could provide.

---

== 3.2 Data Collection & Sample Construction

The study draws on three primary data sources, each capturing a distinct information modality. All data was collected for US-listed public companies involved in completed M&A transactions.

=== 3.2.1 Bloomberg M&A Dataset

The foundational dataset comprises *4,999 completed M&A transactions* announced between August 30, 1994 and December 13, 2022, sourced from the Bloomberg Terminal's Mergers & Acquisitions (MA) function. This 28-year sample window captures multiple market cycles, including the dot-com bubble, the 2008 global financial crisis, and the post-COVID recovery period, providing temporal diversity that strengthens the generalisability of findings.

*Selection criteria* applied during extraction:
- *Geographic scope:* US-listed acquirers (to ensure consistent regulatory environment and disclosure standards)
- *Deal status:* Completed transactions only (to avoid contamination from terminated or pending deals)
- *Data availability:* Deals with available acquirer and target stock price data for CAR computation

Each deal record contains *67 financial variables* spanning eight analytical categories.

=== 3.2.2 SEC EDGAR 10-K Filings

To extract qualitative information about strategic intent and risk awareness, *2,921 annual 10-K filings* were programmatically retrieved from the SEC's EDGAR system.

The EDGAR retrieval pipeline operates as follows:
1. *Ticker-to-CIK Mapping:* Mapped via the SEC's `company_tickers.json` endpoint.
2. *Filing Discovery:* The most recent 10-K filing preceding each deal's announcement date is located.
3. *Full-Text Download:* Complete plain-text formatting downloaded.
4. *Section Parsing:* Extraction of Item 7 (MD&A) and Item 1A (Risk Factors).

=== 3.2.3 Bloomberg Supply Chain (SPLC) Data

Inter-firm supply chain relationships were sourced from the Bloomberg Supply Chain Analysis (SPLC) function. The dataset contains *18,707 unique supplier-customer records* linking acquirer firms to their supply chain partners.

=== 3.2.4 Sample Overlap & Final Dataset Construction

Not all data sources have complete overlap. The merging procedure produces progressively filtered subsets:

#figure(
  table(
    columns: 4,
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    [*Processing Stage*], [*Modality Added*], [*Remaining Deals ($N$)*], [*Feature Space ($RR^d$)*],
    [Raw Bloomberg Extraction], [Financial Base], [4,999], [67],
    [CAR Computation Filter], [Target Variable], [4,510], [68],
    [FinBERT Integration], [Textual (MD&A, RF)], [$sim 2,250$], [196],
    [HeteroGraphSAGE Integration], [Topological], [2,864], [268],
    table.hline(stroke: 0.5pt),
    [*Final Imputed Dataset*], [*Multimodal Fusion*], [*4,999*], [*268*],
    table.hline(stroke: 1pt)
  ),
  caption: [Dataset Construction and Feature Enrichment Pipeline]
)

---

== 3.3 Target Variable: Cumulative Abnormal Returns (CAR)

The target variable for this study is the *Cumulative Abnormal Return (CAR)* over the event window surrounding each deal's announcement date. CAR quantifies the stock market's surprise reaction to the merger announcement.

=== 3.3.1 Market Model Estimation

Abnormal returns are estimated using the single-factor market model, decomposing an acquirer's return into systematic and idiosyncratic components:

$ R_{i,t} = alpha_i + beta_i R_{m,t} + varepsilon_{i,t} $

where:
- $R_{i,t}$ is the return of acquirer $i$ on day $t$
- $R_{m,t}$ is the return of the market index (S&P 500) on day $t$
- $alpha_i$ and $beta_i$ are OLS regression coefficients
- $varepsilon_{i,t}$ is the residual (abnormal return)

#figure(
  align(center)[
    #cetz.canvas({
      import cetz.draw: *
      line((0,0), (10,0), mark: (end: ">", start: "<"))
      
      // Ticks
      line((1, -0.1), (1, 0.1)); content((1, -0.5), [$-260$])
      line((5, -0.1), (5, 0.1)); content((5, -0.5), [$-11$])
      line((7, -0.1), (7, 0.1)); content((7, -0.5), [$-5$])
      line((8, -0.1), (8, 0.1)); content((8, -0.6), [$0$\n(Announcement)])
      line((9, -0.1), (9, 0.1)); content((9, -0.5), [$+5$])
      
      // Brackets
      line((1, 0.3), (1, 0.5), (5, 0.5), (5, 0.3))
      content((3, 0.8), text(size: 8pt)[Estimation Window\n(Market Model: $alpha_i, beta_i$)])
      
      line((7, 0.3), (7, 0.5), (9, 0.5), (9, 0.3))
      content((8, 0.8), text(size: 8pt)[Event Window\n(CAR Computation)])
    })
  ],
  caption: [Event Study Timeline mapping the estimation and event windows relative to $t=0$.]
)

=== 3.3.2 CAR Computation

The abnormal return for day $t$ is computed as:

$ A R_{i,t} = R_{i,t} - (hat(alpha)_i + hat(beta)_i R_{m,t}) $

The Cumulative Abnormal Return over the event window is:

$ C A R_i(tau_1, tau_2) = sum_{t=tau_1}^{tau_2} A R_{i,t} $

=== 3.3.5 Binary Classification Target

For classification tasks, the continuous CAR is binarised:

$ y_i = cases(1 "if" C A R_i > 0 " (positive synergy)", 0 "if" C A R_i <= 0 " (value destruction)") $

---

== 3.4 Feature Engineering Pipeline

=== 3.4.1 Financial Features (M1 Baseline)

The financial feature set comprises the 56 available numerical variables from Bloomberg. Features are imputed via median, standardised, and winsorized:

$ x'_j = (x_j - bar(x)_j) / sigma_{x_j} $

=== 3.4.2 NLP Features --- FinBERT Text Embeddings (M2 Extension)

#figure(
  align(center)[
    #cetz.canvas({
      import cetz.draw: *
      rect((0, 0), (1.5, 1), name: "doc"); content("doc", text(size: 8pt)[10-K\nFiling])
      rect((2.5, 0), (4, 1), name: "chunk"); content("chunk", text(size: 8pt)[512-token\nChunks])
      rect((5, 0), (6.5, 1), name: "bert"); content("bert", text(size: 8pt)[FinBERT\nStack])
      rect((7.5, 0), (9, 1), name: "pool"); content("pool", text(size: 8pt)[Mean\nPooling])
      rect((10, 0), (11.5, 1), name: "pca"); content("pca", text(size: 8pt)[PCA\n$RR^64$])
      
      line("doc.east", "chunk.west", mark: (end: ">"))
      line("chunk.east", "bert.west", mark: (end: ">"))
      line("bert.east", "pool.west", mark: (end: ">"), name: "cls")
      content("cls.top", text(size: 7pt)[[CLS]])
      line("pool.east", "pca.west", mark: (end: ">"))
    })
  ],
  caption: [NLP Representation Learning Pipeline.]
)

Each chunk is passed through the FinBERT encoder, and average pooled:

$ bold(e)_"section" = 1/K sum_{k=1}^K bold(h)_k^"[CLS]" $

=== 3.4.3 Graph Features --- HeteroGraphSAGE Embeddings (M3 Extension)

#figure(
  align(center)[
    #cetz.canvas({
      import cetz.draw: *
      
      // Schema (Left)
      circle((0, 2), radius: 0.3, name: "s1"); content("s1", text(size: 8pt)[S])
      circle((0, 0), radius: 0.3, name: "s2"); content("s2", text(size: 8pt)[S])
      rect((2, 0.5), (3, 1.5), name: "a"); content("a", text(size: 8pt)[A])
      circle((5, 1), radius: 0.3, name: "c"); content("c", text(size: 8pt)[C])
      
      line("s1.east", "a.west", mark: (end: ">"), name: "e1")
      content("e1.top", text(size: 6pt)[supplies])
      line("s2.east", "a.west", mark: (end: ">"))
      
      line("a.east", "c.west", mark: (end: ">", stroke: "dashed"), stroke: (dash: "dashed"), name: "e2")
      content("e2.top", text(size: 6pt)[buys_from])
      
      // Message Passing Math (Right)
      content((8, 1), text(size: 9pt)[$ bold(h)_v^((l+1)) = sigma(bold(W)^((l)) dot "CONCAT"(bold(h)_v^((l)), limits(oplus)_{r in cal(R)} "AGG"_r({bold(h)_u^((l))}))) $])
    })
  ],
  caption: [Heterogeneous Graph Schema and Message Passing aggregation step.]
)

The training objective is self-supervised link prediction:

$ cal(L) = - sum_((u,v) in cal(E)^+) log(sigma(bold(h)_u^top bold(h)_v)) - sum_((u,v') in cal(E)^-) log(1 - sigma(bold(h)_u^top bold(h)_v')) $

---

== 3.5 Multimodal Fusion & Classification

The fused feature vector for deal $i$ is:

$ bold(x)_i = [bold(f)_i^56 || bold(t)_i^128 || bold(g)_i^64 || bold(1)_"has_graph"] in RR^249 $

XGBoost is used as the primary classifier:

$ hat(y)_i^((t)) = hat(y)_i^((t-1)) + f_t(bold(x)_i) $

---

== 3.7 Hypothesis Testing Methodology

#figure(
  table(
    columns: 4,
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    [*Hypothesis*], [*Independent Variable*], [*Dependent Metric*], [*Statistical Test*],
    [*H1: Topological Alpha*], [Configuration (M1 vs. M3)], [5-Fold AUC-ROC], [Paired $t$-test ($p < 0.05$)],
    [*H2: Semantic Divergence*], [Cosine Sim. (MD&A, Risk Factors)], [Continuous CAR], [Bivariate OLS Regression],
    [*H3: Topological Arbitrage*], [Betweenness vs. Clustering Coeff.], [$|C A R|$ Variance], [Levene's Test for Equality],
    table.hline(stroke: 1pt)
  ),
  caption: [Summary of Pre-Registered Hypothesis Testing Framework]
)

=== 3.7.1 H1: The Topological Alpha Hypothesis
Tested via sector stratification and a paired $t$-test.
$ t = bar(d) / (s_d / sqrt(k)) $

=== 3.7.2 H2: The Semantic Divergence Hypothesis
Each deal's cosine similarity to the centroid is computed:
$ "sim"_i^"section" = (bold(e)_i dot bar(bold(e))) / (norm(bold(e)_i) dot norm(bar(bold(e)))) $

A linear model regresses CAR on both similarity scores simultaneously:
$ C A R_i = beta_0 + beta_1 dot "sim"_i^"MDA" + beta_2 dot "sim"_i^"RF" + varepsilon_i $

=== 3.7.3 H3: The Topological Arbitrage Hypothesis
Target nodes are assessed via Betweenness Centrality, Clustering Coefficient, and Degree Centrality.