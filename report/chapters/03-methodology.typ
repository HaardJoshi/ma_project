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


