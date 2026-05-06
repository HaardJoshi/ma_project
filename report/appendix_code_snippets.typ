= Appendix E: Selected Implementation Code Snippets

This appendix presents selected implementation excerpts from the final system rather than a full repository dump. The objective is to document the core engineering logic behind the dissertation's empirical claims in a way that remains readable, auditable, and aligned with the methodology described in Chapter 3.

#pagebreak()

== Appendix Strategy

Not every line of source code is reproduced here. The full project contains data ingestion scripts, model training utilities, evaluation pipelines, dashboard code, configuration files, and visualisation logic; reproducing the entire repository would obscure the methodological contribution rather than clarify it.

The selection criteria used in this appendix are:

- code that directly implements a methodological claim made in the dissertation;
- code that protects against leakage or invalid evaluation;
- code that defines a core modelling component;
- code that supports interpretability or reproducibility.

Accordingly, the appendix focuses on six components:

1. event-study CAR generation;
2. temporal splitting with embargo control;
3. section-aware textual feature extraction;
4. heterogeneous graph embedding construction;
5. multimodal late-fusion prediction;
6. SHAP-based interpretability.

#pagebreak()

== E.1 Two-Stage CAR Pipeline

The first critical implementation block computes Cumulative Abnormal Return (CAR) from market data before any prediction model is trained. This matters because the dissertation explicitly separates label construction from the downstream learning pipeline.

=== Purpose

- Stage 1: estimate realised CAR using an OLS market model over the estimation window;
- Stage 2: train the prediction model using only pre-announcement features.

=== Abridged Code

```python
from scipy.stats import linregress
import numpy as np
import pandas as pd

ESTIMATION_START = -200
ESTIMATION_END = -20
EVENT_START = -5
EVENT_END = 5
MIN_OBS = 120

def compute_car_for_deal(df):
    est = df[(df["relday"] >= ESTIMATION_START) & (df["relday"] <= ESTIMATION_END)]
    evt = df[(df["relday"] >= EVENT_START) & (df["relday"] <= EVENT_END)]

    est = est.dropna(subset=["acq_ret", "mkt_ret"])
    evt = evt.dropna(subset=["acq_ret", "mkt_ret"])

    if len(est) < MIN_OBS or evt.empty:
        return np.nan

    alpha, beta, _, _, _ = linregress(est["mkt_ret"], est["acq_ret"])
    evt = evt.copy()
    evt["expected_ret"] = alpha + beta * evt["mkt_ret"]
    evt["abnormal_ret"] = evt["acq_ret"] - evt["expected_ret"]

    return evt["abnormal_ret"].sum()

final_df["carm5p5"] = final_df.groupby("deal_id").apply(compute_car_for_deal)
final_df["car_binary"] = (final_df["carm5p5"] > 0).astype(int)
```

=== Why This Snippet Matters

This excerpt shows that the target variable is generated from a conventional event-study process before the classifier or regressor is trained. It also makes clear that the binary target used in the classification pipeline is derived directly from the sign of CAR, not from post-hoc labelling.

#pagebreak()

== E.2 Temporal Splitting and Embargo Control

A core validity claim in the dissertation is that the pipeline prevents forward-looking leakage through strict temporal splitting and an 11-trading-day embargo between partitions.

=== Purpose

- preserve chronological realism;
- prevent overlapping CAR event windows across train, validation, and test sets;
- ensure preprocessing is fitted only on training data.

=== Abridged Code

```python
import pandas as pd

EMBARGO_DAYS = 11


def apply_temporal_split(df):
    train = df[df["announce_year"].between(2000, 2016)].copy()
    valid = df[df["announce_year"].between(2017, 2019)].copy()
    test = df[df["announce_year"].between(2020, 2023)].copy()
    return train, valid, test


def embargo_boundary(left_df, right_df, left_max_date, right_min_date):
    left_keep = left_df[left_df["announce_date"] < left_max_date - pd.Timedelta(days=EMBARGO_DAYS)]
    right_keep = right_df[right_df["announce_date"] > right_min_date + pd.Timedelta(days=EMBARGO_DAYS)]
    return left_keep, right_keep

train, valid, test = apply_temporal_split(df)
train, valid = embargo_boundary(train, valid, train["announce_date"].max(), valid["announce_date"].min())
valid, test = embargo_boundary(valid, test, valid["announce_date"].max(), test["announce_date"].min())
```

=== Why This Snippet Matters

This code operationalises one of the dissertation's strongest methodological safeguards. Without this step, neighbouring deals can share overlapping return windows and silently contaminate evaluation.

#pagebreak()

== E.3 Section-Aware FinBERT Extraction

The text pipeline deliberately separates Management Discussion and Analysis (MDA) from Risk Factors rather than treating the filing as a single undifferentiated document.

=== Purpose

- preserve section-specific semantics;
- avoid signal cancellation between strategic-fit language and liability language;
- compress document embeddings into a stable downstream representation.

=== Abridged Code

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def chunk_text(text, max_len=512, stride=256):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    windows = []
    for i in range(0, max(1, len(tokens)), stride):
        chunk = tokens[i:i + max_len - 2]
        if not chunk:
            break
        windows.append([tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id])
        if i + max_len - 2 >= len(tokens):
            break
    return windows


def embed_section(text):
    windows = chunk_text(text)
    vecs = []
    with torch.no_grad():
        for win in windows:
            x = torch.tensor([win])
            out = model(x, output_hidden_states=True)
            cls_vec = out.hidden_states[-2][0, 0, :].cpu().numpy()
            vecs.append(cls_vec)
    return np.mean(vecs, axis=0)

acq_mda_vec = embed_section(acq_item7_text)
acq_rf_vec = embed_section(acq_item1a_text)
```

=== Why This Snippet Matters

This excerpt captures the exact modelling principle behind the Semantic Divergence Hypothesis: the two filing sections are processed independently and only merged later, rather than collapsed at source.

#pagebreak()

== E.4 Heterogeneous Graph Construction and Embedding

The graph stream is the architectural feature that differentiates the final system from standard tabular M&A prediction pipelines.

=== Purpose

- encode firm position inside a supply-chain network;
- distinguish upstream and downstream relationships through typed edges;
- extract firm-level embeddings for downstream fusion.

=== Abridged Code

```python
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
import torch.nn.functional as F
import torch


def build_hetero_graph(node_x, supplies_edges, buysfrom_edges):
    data = HeteroData()
    data["company"].x = node_x
    data[("company", "supplies", "company")].edge_index = supplies_edges
    data[("company", "buysfrom", "company")].edge_index = buysfrom_edges
    return data


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=64):
        super().__init__()
        self.conv1 = HeteroConv({
            ("company", "supplies", "company"): SAGEConv(in_dim, hidden_dim),
            ("company", "buysfrom", "company"): SAGEConv(in_dim, hidden_dim),
        }, aggr="mean")
        self.conv2 = HeteroConv({
            ("company", "supplies", "company"): SAGEConv(hidden_dim, out_dim),
            ("company", "buysfrom", "company"): SAGEConv(hidden_dim, out_dim),
        }, aggr="mean")

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict["company"]
```

=== Why This Snippet Matters

This is the clearest code-level expression of the Topological Alpha argument. Separate message-passing weights are learned for different relationship types, which would be impossible in a flat financial table.

#pagebreak()

== E.5 Late-Fusion Prediction Head

The final predictive system uses late fusion rather than end-to-end joint training of all modalities.

=== Purpose

- keep each modality independently encoded;
- permit controlled ablations by enabling or disabling individual streams;
- produce a single fused representation for classification or regression.

=== Abridged Code

```python
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fin_proj = ProjectionHead(56, 64)
        self.txt_proj = ProjectionHead(128, 64)
        self.gph_proj = ProjectionHead(64, 32)
        self.classifier = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, fin_x, txt_x, gph_x):
        z_fin = self.fin_proj(fin_x)
        z_txt = self.txt_proj(txt_x)
        z_gph = self.gph_proj(gph_x)
        z = torch.cat([z_fin, z_txt, z_gph], dim=1)
        return self.classifier(z)
```

=== Why This Snippet Matters

This excerpt shows the central design decision of the dissertation: modality-specific extraction first, multimodal combination second. It is also the component that makes the ablation ladder methodologically clean.

#pagebreak()

== E.6 SHAP-Based Interpretation

The dissertation does not treat interpretability as optional presentation. SHAP is used to test whether the multimodal lift is economically credible at the feature level.

=== Purpose

- quantify feature-level contribution to prediction output;
- verify that graph and text signals are genuinely used by the classifier;
- generate visual evidence for the SHAP appendix figures and dashboard.

=== Abridged Code

```python
import shap
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

model = XGBClassifier(eval_metric="auc", random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
mean_abs = np.abs(shap_values).mean(axis=0)

importance = pd.DataFrame({
    "feature": X_train.columns,
    "mean_abs_shap": mean_abs,
}).sort_values("mean_abs_shap", ascending=False)

importance.head(15)
```

=== Why This Snippet Matters

This is the implementation bridge between the predictive model and the interpretability claims made in Chapter 4. It demonstrates how the study moves from raw classifier output to ranked economic drivers.

#pagebreak()

== E.7 Recommended Presentation Principles

For an appendix of this type, the best practice is not to include every source file. A selective appendix is clearer and academically stronger when it does the following:

- include only code that directly supports a dissertation claim;
- trim boilerplate, imports, logging, and helper functions unless they are methodologically important;
- prefer 20--40 line abridged excerpts over full scripts;
- label each snippet by purpose, input, output, and chapter linkage;
- keep the full repository available separately for reproducibility.

In other words, the appendix should function as an explanatory engineering record, not as a raw code dump.

== E.8 Suggested Mapping to Chapters

#table(
  columns: (1.7fr, 1.2fr, 2.6fr),
  inset: 8pt,
  stroke: 0.4pt + gray,
  [*Appendix Section*], [*Links to Chapter*], [*Why It Belongs*],

  [E.1 Two-Stage CAR Pipeline], [3.8], [Implements the target variable and shows that labels are derived before model training.],
  [E.2 Temporal Splitting and Embargo], [3.5.3], [Documents the main leakage-control mechanism.],
  [E.3 Section-Aware FinBERT], [3.6.2], [Supports the semantic-divergence argument and M2 reversal discussion.],
  [E.4 HeteroGraphSAGE], [3.6.3], [Shows how supply-chain topology is encoded.],
  [E.5 Late Fusion], [3.7.2], [Documents the multimodal architecture and ablation logic.],
  [E.6 SHAP Interpretation], [4.6.1], [Connects the model to the interpretability results.],
)

== E.9 Note on Repository Availability

The appendix is intentionally selective. Full implementation files, configuration artefacts, and dashboard code are maintained in the project repository and can be supplied separately if examiners require full computational auditability.
