#let AR = "AR"
#let CAR = "CAR"

= Methodology

== Introduction

This chapter details the research design, architectural implementation, and evaluation proto-
cols employed to construct the proposed Heterogeneous Graph Neural Network (HGNN). The
methodology is structured to operationalize the theoretical findings of Chapter 2, translating
the need for ”multimodal fusion” into a rigorous engineering specification.
It begins by defining the Research Framework, justifying the selection of a Quantitative Methodology and an Experimental Prototyping SDLC to address the non-deterministic nature of deep learning. Next, it outlines the Tools and Technologies, specifying the computational stack (PyTorch, Hugging Face) required for high-dimensional tensor processing. The core of the methodology is presented in the Implementation Architecture and Data Ingestion, which define the ”Dual-Stream” system architecture and the strict ”pre-merger windowing” strategy used to prevent data leakage. Finally, it details the Evaluation Protocol, establishing the hierarchy of baselines (Logistic Regression to XGBoost) used to validate the system’s performance, and the lastly it addresses the technical limitations and ethical considerations inherent in automated financial modeling.

== System Architecture and Mathematical Formulation

The core objective of this study is to construct a predictive framework that transcends the limitations of tabular M&A models by treating each merger not as an isolated transaction, but as a topological event within a complex industrial ecosystem @3.1Newman_2003. To achieve this, the study proposes a dual-stream Heterogeneous Graph Neural Network (HGNN) that operationalises the hypothesis that post-merger synergy $S$ is a latent variable conditioned on three distinct but complementary modalities @3.1Baltrušaitis-Ahuja: firms' financial capacity $F$, the semantic intent of their disclosures $T$, and their structural position within the network $G$. Synergy is observed through Cumulative Abnormal Return (CAR) around the announcement, following the standard event-study market-model framework. This architecture ensures that the prediction is based on a temporally aligned, multimodal state space, recovering the signal lost by traditional monomodal models—directly addressing the "Topological Blindness" and "Wrong Target" critiques identified in the literature review.

=== Problem Formulation
We formulate the synergy prediction task as a supervised regression problem. Let $cal(D) = {(G_(t-Delta)^((i)), T_(t-Delta)^((i)), F_(t-Delta)^((i)), y_i)}_(i=1)^(N)$ denote the sample of $N$ completed M&A deals, where each deal $d_i$ corresponds to an acquirer-target pair announced at time $t$. To strictly prevent look-ahead bias @3.1COHEN_FRAZZINI_2008, all input features are frozen at the most recent fiscal reporting period $t-Delta$ prior to the event, preventing any post-announcement information from leaking into the model.

The predictive task is to learn a parametric function $f_theta$ such that:

$ hat(y)_i = f_theta (G_(t-Delta)^((i)), T_(t-Delta)^((i)), F_(t-Delta)^((i))) $

where $hat(y)_i$ is the model's estimate of the deal's synergy.

The target variable $y_i in RR$ is the *Cumulative Abnormal Return (CAR)* over a symmetric event window $[-5, +5]$ days. Following standard event study methodology @3.1Mackinlay1997EventSI, CAR is computed using the market model estimated over a pre-event estimation window:

$ R_(i t) = alpha_i + beta_i R_(m t) + epsilon_(i t) $

where $R_(i t)$ is the return of firm $i$ at time $t$ and $R_(m t)$ is the market return. Abnormal returns ($AR$) are then calculated as the residual:

$ AR_(i t) = R_(i t) - (hat(alpha)_i + hat(beta)_i R_(m t)) $

Finally, CAR is obtained by aggregating these residuals over the event window $cal(T)$:

$ CAR_i = sum_(t in cal(T)) AR_(i t) $

In this framework, synergy $S$ is treated as a latent construct for which $CAR_i$ serves as an observable proxy.

The model is trained as a supervised regression system by minimizing the Mean Squared Error (MSE) with L2 regularization:

$ cal(L)(theta) = 1/N sum_(i=1)^(N) (y_i - hat(y)_i)^2 + lambda ||theta||_2^2 $

While stock returns are known to exhibit heavy-tailed distributions, MSE remains the primary objective for comparability with prior financial prediction work. However, robust alternatives such as the Huber loss are considered in sensitivity analyses to mitigate the effect of extreme return outliers.

=== Dual-Stream HGNN Architecture
The function $f_theta$ is implemented as a dual-stream neural network designed to fuse topological, semantic, and financial information (Figure 3.1).

+ *Stream A: The Topological Encoder (GraphSAGE).* This stream processes the pre-merger ecosystem snapshot $G_(t-Delta)$, which encodes firms as nodes with heterogeneous edges (e.g., supplier-customer, competitors). It utilizes a Heterogeneous GraphSAGE operator to sample and aggregate neighbor features @3.1HamiltonYL17. This inductive approach allows the model to generate structural embeddings ($h_"struct"$) for unseen firms in evolving financial graphs, capturing network centrality and supply chain dependency risks.

+ *Stream B: The Semantic Encoder (FinBERT).* This stream processes the unstructured textual disclosures $T_(t-Delta)$ (specifically 10-K filings). It employs *FinBERT* (Araci, 2019), a BERT-based model pre-trained on large financial corpora, to extract "semantic embeddings" ($h_"text"$). For each relevant section, the `[CLS]` token representation from the penultimate layer is taken as the document-level embedding. Crucially, FinBERT's approximately 110M parameters are *frozen* during training to prevent overfitting, given the limited sample size of M&A deals.

+ *The Fusion Module.* A financial-feature embedding $h_"fin"$ is obtained by passing the standardized raw financial ratios $F_(t-Delta)$ through a shallow linear projection layer. The three distinct embeddings are then concatenated into a single high-dimensional vector $Z = [h_"struct" || h_"text" || h_"fin"]$. This vector is passed through a Multi-Layer Perceptron (MLP) with non-linear activation functions (ReLU) and Dropout layers to project the multimodal signal onto the scalar output $hat(y)$ (CAR).

$ R_{i t} = alpha_i + beta_i R_{m t} + epsilon_{i t} $


---

Part II: Methodology Chapter
3. Methodology
3.1 Research Philosophy and Design
This study adopts a post-positivist epistemological stance, treating M&A synergy as a latent, probabilistic construct that can be approximated through empirical measurement of market reactions and structured firm relationships. While acknowledging that financial markets are not perfectly efficient, the research operates within the semi-strong form of the Efficient Market Hypothesis (Fama, 1970), wherein publicly available information — including financial fundamentals, regulatory filings, and inter-firm network topology — is treated as a viable predictor signal. The overarching research design is quantitative and deductive: three a priori hypotheses (H1: Topological Alpha, H2: Semantic Divergence, H3: Topological Arbitrage) are specified before analysis and tested through controlled ablation experiments.

The study employs a cross-sectional observational design. Since M&A deals are historical and non-repeatable, no experimental manipulation is possible; instead, causal inference is approximated through systematic covariate control, ablation modelling, and statistical hypothesis testing over a large deal sample. This design choice reflects standard practice in empirical corporate finance (MacKinlay, 1997).

3.2 Research Hypotheses
Three hierarchical hypotheses structure the empirical programme:

H1 (Topological Alpha Hypothesis): Supply-chain and competitor network centrality metrics derived from SPLC data carry statistically significant predictive signal for acquirer CAR, incremental to financial fundamentals alone.

H2 (Semantic Divergence Hypothesis): The cosine distance between acquirer and target FinBERT embeddings of their respective 10-K MD&A sections is a significant predictor of post-acquisition CAR, capturing strategic fit information not encoded in accounting ratios.

H3 (Topological Arbitrage Hypothesis): The full multimodal fusion of financial, textual, and graph features significantly outperforms all single-modality baselines on held-out CAR prediction, as measured by MAE and R².

3.3 Data Sources and Collection
3.3.1 M&A Deal Universe
The primary dataset is sourced from the London Stock Exchange Group (LSEG) Refinitiv database, which provides comprehensive deal-level financial attributes for completed M&A transactions. Five raw CSV exports are merged into a unified dataset via scripts/data/build_combined_dataset.py, producing data/interim/ma_combined.csv. The deal universe is restricted to: completed acquisitions of publicly listed targets by publicly listed US acquirers; transactions between 2000–2023; deal values exceeding USD 50 million (to ensure sufficient market microstructure data for CAR estimation). These filters follow established practice in the empirical M&A literature (Betton et al., 2008) and ensure a minimum of 120 trading days in the estimation window.

3.3.2 Equity Return Data
Daily equity returns for acquirer firms and benchmark (S&P 500 index) are retrieved via the Bloomberg Terminal using custom ticker-matching logic in scripts/data/pull_car_data.py and scripts/data/merge_bbg_data.py. Returns are aligned to deal announcement dates and stored in a long-format time series (timeseries_long.csv) with a rel_day field denoting days relative to the announcement (day 0). Failed ticker lookups are retried with fuzzy-matching heuristics via scripts/data/retry_failed_tickers.py.

3.3.3 Textual Data
10-K annual filings for acquirer and target firms are retrieved from the SEC EDGAR full-text search API, targeting the MD&A (Item 7) and Risk Factors (Item 1A) sections for the fiscal year immediately preceding the announcement. Extraction is handled by scripts/features/ text pipeline scripts. Documents are processed into FinBERT [CLS] token embeddings as described in Section 3.5.2.

3.3.4 Supply Chain Network Data
Inter-firm supply chain and competitor relationships are sourced from Bloomberg SPLC (Supply Chain Analysis), which maps disclosed customer-supplier relationships and key competitors for publicly listed firms. The SPLC data is merged with the deal universe via scripts/data/merge_splc_data.py, matching on Bloomberg ticker symbols. This forms the edge set for the heterogeneous graph constructed in Section 3.5.3.

3.4 Data Preprocessing
3.4.1 Cleaning and Quality Control
Raw LSEG exports undergo systematic cleaning in scripts/data/data_cleaning.py, which handles: (a) date parsing and standardisation across inconsistent LSEG date formats (resolved via scripts/data/fix_dates.py); (b) deduplication of deal records sharing the same acquirer-target-announcement-date triplet; (c) removal of records with missing acquirer ticker or announcement date; and (d) currency normalisation to USD using period-end exchange rates. Data quality is verified via scripts/data/verify_sources.py.

3.4.2 Feature Engineering and Normalisation
Financial features comprise 50+ ratio-level variables sourced from LSEG, spanning acquirer and target leverage, liquidity, profitability, and deal structure characteristics. The preprocessing pipeline in scripts/data/data_processing.py applies: (a) Winsorisation at the 1st and 99th percentile to bound the influence of outliers, following standard practice in accounting research; (b) z-score standardisation (zero mean, unit variance) computed on training-set statistics only and applied to validation and test sets to prevent data leakage; (c) stratified train/validation/test splitting (70/15/15) by announcement year to prevent temporal leakage.

3.4.3 Missing Data Strategy
Features with >40% missing values are excluded. For remaining missing values, median imputation is applied for continuous features and mode imputation for categorical indicators. Imputation statistics are fitted on the training set only.

3.5 Feature Extraction
3.5.1 Block A — Financial Features
The financial feature vector 
h
F
∈
R
d
F
h 
F
​
 ∈R 
d 
F
​
 
  is constructed directly from the standardised preprocessing output. For baseline models (Ridge Regression, ElasticNet, XGBoost), 
h
F
h 
F
​
  is used directly. For the MLP and fusion models, it passes through a ProjectionHead (linear layer + ReLU) that maps 
h
F
h 
F
​
  to a lower-dimensional embedding 
h
^
F
∈
R
64
h
^
  
F
​
 ∈R 
64
  before concatenation.

3.5.2 Block B — Textual Features (FinBERT)
Each firm's MD&A and Risk Factors text is tokenised and passed through the pre-trained FinBERT model (Araci, 2019) — a BERT-base architecture fine-tuned on financial communications corpora. The [CLS] token embedding from the final transformer layer is used as the document representation 
h
T
∈
R
768
h 
T
​
 ∈R 
768
 . FinBERT weights are frozen during all downstream training to prevent overfitting on the relatively small deal sample; only the downstream projection head is trained. For H2 testing, the cosine distance between acquirer and target embeddings is computed as:

SemanticDiv
i
=
1
−
h
T
acq
⋅
h
T
tgt
∥
h
T
acq
∥
⋅
∥
h
T
tgt
∥
SemanticDiv 
i
​
 =1− 
∥h 
T
acq
​
 ∥⋅∥h 
T
tgt
​
 ∥
h 
T
acq
​
 ⋅h 
T
tgt
​
 
​
 
This scalar divergence score is included as an additional feature for H2 ablation experiments.

3.5.3 Block C — Graph Features (GraphSAGE on HGNN)
The inter-firm network is constructed as a heterogeneous graph 
G
=
(
V
,
E
,
T
v
,
T
e
)
G=(V,E,T 
v
​
 ,T 
e
​
 ), where node types 
T
v
T 
v
​
  include firm, sector, and country, and edge types 
T
e
T 
e
​
  include supplier_of, customer_of, competitor_of, and acquires. Graph construction is handled by scripts/graphs/build_hetero_graph.py using PyTorch Geometric's HeteroData object.

Node-level features are initialised with degree centrality, betweenness centrality, and the standardised financial feature vector. GraphSAGE (Hamilton et al., 2017) is then applied with mean aggregation over two message-passing layers to produce node embeddings 
h
G
∈
R
d
G
h 
G
​
 ∈R 
d 
G
​
 
 , which encode each firm's structural position within the ecosystem. For deals without SPLC coverage, 
h
G
h 
G
​
  is set to a zero vector and the graph stream is masked in the fusion model via the has_graph flag.

3.6 Model Architecture
3.6.1 Baseline Models
Four baselines are trained on Block A features only: a naïve mean predictor (lower bound), Ridge Regression, ElasticNet (for feature selection insight), and XGBoost (to capture non-linear financial interactions). These establish the performance ceiling achievable from financial data alone and provide the comparative baseline for H1, H2, and H3.

3.6.2 Fusion Model
The primary model is a late-fusion multimodal architecture (src/models/fusion.py). Each active stream passes through its own ProjectionHead, and the resulting embeddings are concatenated:

z
i
=
[
h
F
∥
h
T
∥
h
G
]
∈
R
d
F
′
+
d
T
′
+
d
G
′
z 
i
​
 =[h 
F
​
 ∥h 
T
​
 ∥h 
G
​
 ]∈R 
d 
F
′
​
 +d 
T
′
​
 +d 
G
′
​
 
 
where 
d
F
′
=
64
d 
F
′
​
 =64, 
d
T
′
=
64
d 
T
′
​
 =64, 
d
G
′
=
32
d 
G
′
​
 =32 by default. The concatenated vector 
z
i
z 
i
​
  is passed through a two-layer MLP prediction head with ReLU activation and dropout (
p
=
0.3
p=0.3) to produce the scalar CAR prediction 
y
^
i
y
^
​
  
i
​
 . The modular design allows any subset of streams to be disabled, enabling the controlled ablation experiments needed for hypothesis testing.

3.6.3 Training Configuration
All PyTorch models are trained with: AdamW optimiser, learning rate scheduled via cosine annealing with warm restarts; MSE loss as the primary objective; early stopping on validation MAE with a patience of 15 epochs; batch size of 64; and a fixed random seed (set via set_seed() in src/training/trainer.py) for reproducibility. Device selection (CUDA / Apple MPS / CPU) is handled automatically via src/config.py.

3.7 Target Variable: Cumulative Abnormal Return
The target variable 
y
i
y 
i
​
  for each deal is the Cumulative Abnormal Return over the event window [-5, +5] trading days relative to announcement date, computed via the market model (Brown & Warner, 1985). Formally:

CAR
i
=
∑
t
=
−
5
+
5
A
R
i
,
t
where
A
R
i
,
t
=
R
i
,
t
−
(
α
^
i
+
β
^
i
R
m
,
t
)
CAR 
i
​
 = 
t=−5
∑
+5
​
 AR 
i,t
​
 whereAR 
i,t
​
 =R 
i,t
​
 −( 
α
^
  
i
​
 + 
β
^
​
  
i
​
 R 
m,t
​
 )
The OLS parameters 
α
^
i
,
β
^
i
α
^
  
i
​
 , 
β
^
​
  
i
​
  are estimated over a 120–240 trading day window ending 10 days before announcement ([-250, -10]), requiring a minimum of 120 valid observations to be included in the sample. This follows MacKinlay (1997) and is implemented in scripts/data/compute_car.py.

3.8 Hypothesis Testing
Each hypothesis is tested through model ablation combined with statistical significance testing:

H1: Compare financial_graph.yaml vs. financial_only.yaml. A paired Diebold-Mariano test on hold-out prediction errors assesses whether the graph stream yields statistically significant MAE improvement.

H2: A Pearson/Spearman correlation test between SemanticDiv_i and CAR is first run. Then financial_text.yaml vs. financial_only.yaml ablation is evaluated using the same DM test.

H3: full_fusion.yaml is compared against all single-modality baselines. Effect size (Cohen's d on hold-out error distributions) and R² improvement are reported alongside p-values.

All tests use a significance threshold of 
α
=
0.05
α=0.05 with Bonferroni correction applied across the three hypothesis tests to control the family-wise error rate.

3.9 Evaluation Metrics
Primary metrics are Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² (coefficient of determination), all computed on the held-out test set. MAE is the primary metric given its interpretability in percentage-point CAR terms. Secondary evaluation includes a directional accuracy metric (proportion of deals where the predicted CAR sign matches the actual sign), which has practical relevance for deal advisory applications.

3.10 Ethical Considerations and Limitations
All data used is commercially licensed (LSEG, Bloomberg) and contains no personally identifiable information. The study does not involve human participants. Key methodological limitations include: (a) the SPLC network captures only disclosed relationships, potentially biasing graph features toward larger firms with more reporting obligations; (b) frozen FinBERT embeddings may not fully capture M&A-specific language not present in the FinBERT training corpus; (c) the OLS market model assumes stationarity of beta over the estimation window, which may be violated for firms undergoing strategic repositioning pre-deal; (d) the sample is restricted to US listed firms, limiting generalisability to cross-border or private-equity transactions.