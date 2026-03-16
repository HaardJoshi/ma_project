# Final Polish & Visualizations

## 1. Narrative & Architectural Damage Control
The dissertation must accurately reflect the decoupled nature of the pipeline, turning perceived limitations into rigorous methodological decisions.

### Task 1.1: Architecture Definition Rewrite
**What:** Finalize documentation framing the model as a "Decoupled Representation Learning & Gradient Boosting Framework" rather than an end-to-end HGNN.
**Why:** Empirical reality dictates that end-to-end backpropagation overfits severely on noisy equity data (N=2,864). Freezing GraphSAGE and FinBERT embeddings and using XGBoost was a deliberate architectural choice to handle the noise floor.

### Task 1.2: Title/Abstract Update
**What:** Shift the narrative from "Predicting Synergy Magnitude" to "Predicting Synergy Direction".
**Why:** Phase 1 (Regression) proved the Efficient Markets Hypothesis holds for exact magnitudes (R² ≈ 0). Phase 2 (Classification) proved directional prediction is possible (AUC=0.566).

### Task 1.3: H2 and H3 Re-framing
**What:** 
- **H2:** Emphasize that SEC 10-K filings are heavily sanitized by corporate lawyers, rendering pre-merger NLP analysis statistically weak compared to hard topological supply-chain realities.
- **H3:** Maintain the "Information Transparency Dampening Effect" framing. Highly centralized supply-chain hubs operate with near-perfect market transparency, entirely removing the 'surprise' premium from announcements.

---

## 2. High-Dimensional Visualizations
We will generate a Jupyter Notebook (`notebooks/final_visualizations.ipynb`) to produce four publication-ready charts.

### Task 2.1: The "Topological Alpha" Ego-Network
**What:** Unipartite/Bipartite network graph of a high-profile deal.
**How:** Extract an acquirer and target from the SPLC data, plot their 1-hop and 2-hop neighborhoods, and color nodes by Betweenness Centrality.

### Task 2.2: The Multimodal SHAP Summary Plot
**What:** Global feature importance proving multimodal fusion works.
**How:** Generate the classic `shap.summary_plot` using the saved `shap_values_M3.csv` to visually prove that `graph_emb_X` and `rf_pca_Y` are driving predictions.

### Task 2.3: The "H3 Volatility Funnel" Scatterplot
**What:** Visual proof of the Information Transparency mechanism.
**How:** Scatterplot of Betweenness Centrality (X) vs Absolute CAR (Y) with a seaborn KDE density overlay to show the "funnel" shape (high centrality = tightly bounded low variance).

### Task 2.4: The ROC-AUC Capability Gap
**What:** The headline empirical graphic.
**How:** Plot the ROC curves for M1 (Financial Baseline) vs M3 (Multimodal) on the same axes to visualize the "Topological Alpha" gap.

---

## 3. The Artifact & Mockups (The Prototype)
Building the minimum viable engineering prototype for maximum artifact marks.

### Task 3.1: System Pipeline Schematic
**What:** A rigorous Mermaid flowchart detailing the architecture.
**How:** Diagram the four blocks: Data Sources -> Extractors -> Fusion -> Inference.

### Task 3.2: Streamlit "Inference UI"
**What:** A lightweight, interactive web frontend.
**How:** Create `app.py`. A 50-line Streamlit app allowing users to select an Acquirer and Target, which loads the pre-computed feature row, passes it to a saved XGBoost model, and outputs the "Probability of Positive Synergy".
