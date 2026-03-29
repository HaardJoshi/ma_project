"""
4_Methodology_Engine.py  —  Interactive Pipeline DAG
=====================================================
Turn 2: Full rewrite with:
  - col_nav [1] | col_viz [2.5] layout
  - st.radio with explicit integer index extraction
  - stage_idx injected as JS literal: const ACTIVE_STAGE = {stage_idx};
  - Vanilla JS / SVG animated node-link DAG rendered via st.components.v1.html
  - st.expander blocks for source code + stage output statistics
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import numpy as np

from utils import setup_page

setup_page(title="Methodology Engine")

st.markdown(
    "<h1><span style='color:#00FFAA;'>Phase 2:</span> Methodology Engine</h1>",
    unsafe_allow_html=True,
)
st.caption("Interactive pipeline DAG — click a stage to inspect architecture, source, and output statistics.")
st.markdown("---")

# ─── Stage definitions ────────────────────────────────────────────────────────
options = [
    "1. Data Ingestion",
    "2. Target Formulation",
    "3. Representation Learning",
    "4. Multimodal Fusion",
    "5. Inference",
]

STAGE_META = {
    0: {
        "title": "Step 1 · Data Ingestion",
        "desc": """
The foundational feature matrix is assembled across two parallel extraction APIs:

1. **Bloomberg Desktop API (`BDP / BDH`)** — 56 fundamental financial ratios: margins, leverage, P/E, EBITDA multiples, R&D ratios, and trailing 12-month metrics for both acquirer and target.
2. **SEC EDGAR Pipeline** — fuzzy CUSIP/ticker matching fetches the 10-K filings for every deal. Item 7 (MD&A) and Item 1A (Risk Factors) are extracted as raw text.
3. **Bloomberg SPLC** — supply-chain relationship links (supplier/buyer edges) between 5,730 unique corporate entities.

The combined raw dataset spans **4,999 deals across 28 fiscal years** before filtering.
""",
        "sources": ["scripts/pull_car_data.py", "scripts/generate_bbg_excel.py",
                    "scripts/generate_splc_excel.py", "scripts/run_edgar_fetch.py",
                    "scripts/fix_dates.py", "scripts/retry_failed_tickers.py",
                    "scripts/merge_bbg_data.py", "scripts/merge_splc_data.py"],
        "stats": {"Raw deals": "4,999", "Date range": "1993 – 2023",
                  "Financial features": "56", "SPLC edges": "7,582",
                  "Post-filter deals": "2,864", "NaN imputation": "Median (5-fold stable)"},
    },
    1: {
        "title": "Step 2 · Target Formulation (CAR)",
        "desc": r"""
To construct the ground-truth synergy label, we apply the **Market Model Event Study**:

$$E[R_{it}] = \alpha_i + \beta_i \cdot R_{mt} + \epsilon_{it}$$

$$AR_{it} = R_{it} - E[R_{it}]$$

$$CAR_i = \sum_{t=-5}^{+5} AR_{it}$$

Where $\beta_i$ is estimated over a 250-day estimation window prior to the announcement. The CAR captures the **exogenous market surprise** attributable solely to the deal announcement.

**Binary encoding:** $y_i = \mathbf{1}[CAR_i > 0]$ ← Class 1 (value-creating)
""",
        "sources": ["scripts/compute_car.py", "scripts/verify_car.py",
                    "scripts/run_preprocessing.py"],
        "stats": {"Event window": "[-5, +5] days", "Estimation window": "250 days",
                  "Class 1 (CAR > 0)": "1,260 deals (44.0%)",
                  "Class 0 (CAR ≤ 0)": "1,604 deals (56.0%)",
                  "CAR mean": "-0.0127", "CAR std": "0.0939"},
    },
    2: {
        "title": "Step 3 · Representation Learning (FinBERT + GraphSAGE)",
        "desc": """
Two complementary deep-learning encoders generate the M2 and M3 modalities:

**FinBERT (NLP):**  ProsusAI's 110M-parameter transformer, pre-trained on 4.9B financial tokens, encodes MD&A and Risk Factor text as 768-dimensional contextual embeddings. PCA then compresses each to **64 principal components** (128 total), retaining >90% of linguistic variance.

**HeteroGraphSAGE (Graph):**  A 2-layer inductive neighbourhood aggregator operating on the PyG `HeteroData` object with edge types `(company, supplies, company)` and `(company, buys_from, company)`. Each acquirer's node embedding is projected to **64 dimensions** via mean-aggregation over 1-hop and 2-hop supplier neighbours.
""",
        "sources": ["scripts/run_text_features.py", "scripts/build_hetero_graph.py",
                    "scripts/train_hetero_graph.py", "scripts/merge_hetero_embeddings.py"],
        "stats": {"FinBERT params": "110M", "MD&A PCA dims": "64",
                  "Risk Factor PCA dims": "64", "Graph nodes": "5,730",
                  "Graph edges": "7,582", "GraphSAGE dims": "64",
                  "Text-coverage deals": "1,279"},
    },
    3: {
        "title": "Step 4 · Multimodal Fusion",
        "desc": """
All modalities are horizontally concatenated into a single feature matrix before the classifier:

| Modality | Source | Dimensions |
|---|---|---|
| M1: Financial | Bloomberg BDP/BDH | 56 |
| M2: NLP Text | FinBERT → PCA | 128 |
| M3: Graph | HeteroGraphSAGE | 64 |
| **Total (M3)** | | **248** |

The fusion vector is passed into **XGBoost** (`binary:logistic`, scale_pos_weight-corrected for 56:44 imbalance) and an **MLP** (2 hidden layers, ReLU, dropout 0.3). A 5-fold stratified cross-validation loop estimates AUC-ROC without leakage.
""",
        "sources": ["scripts/build_combined_dataset.py", "scripts/train_classifier.py",
                    "scripts/train_classifier_v2.py", "scripts/tune_xgboost.py"],
        "stats": {"M1 AUC (XGBoost)": "0.5408", "M2 AUC (XGBoost)": "0.5289",
                  "M3 AUC (XGBoost)": "0.5655 ✅", "Tuning": "Optuna (100 trials)",
                  "CV folds": "5 (stratified)", "Pos-weight": "1.273"},
    },
    4: {
        "title": "Step 5 · Inference & Explainability",
        "desc": """
The trained M3 ensemble produces per-deal **synergy probability scores** that are:

1. **SHAP-decomposed** — TreeExplainer isolates each feature's marginal contribution, enabling auditable deal-level explanations.
2. **Hypothesis-tested** — Three structural hypotheses (H1: topological alpha, H2: semantic divergence, H3: variance compression) are validated with paired t-tests and Levene's F-test on held-out folds.
3. **Dashboard-rendered** — The 10 extremal deals (top/bottom 5 CAR) stream live SHAP waterfalls and ego-network visualisations into the Deal Terminal.
""",
        "sources": ["scripts/test_h1.py", "scripts/test_h2.py", "scripts/test_h3.py",
                    "generate_payloads.py", "generate_macro_stats.py"],
        "stats": {"H1 SC Δ-AUC": "+0.059 (p=0.005 ✅)", "H2 β_MDA": "+0.0044 ✅",
                  "H2 β_RF": "-0.0080 ✅", "H3 Levene p": "0.008 ✅",
                  "SHAP features": "248", "Deals scored": "2,864"},
    },
}

# ─── Layout ───────────────────────────────────────────────────────────────────
col_nav, col_viz = st.columns([1, 2.5])

with col_nav:
    selected = st.radio("", options, label_visibility="collapsed")
    stage_idx = options.index(selected)          # explicit integer index

    st.markdown("<br>", unsafe_allow_html=True)

    meta = STAGE_META[stage_idx]

    st.markdown(f"#### {meta['title']}")
    st.markdown(meta["desc"])

with col_viz:
    # ── JS DAG ──────────────────────────────────────────────────────────────
    dag_html = f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin:0; background:#0E1117; font-family:'Courier New', monospace; }}
  svg {{ width:100%; height:480px; }}
  .node-circle {{ stroke-width:2.5px; cursor:pointer; transition: all 0.25s ease; }}
  .node-circle:hover {{ filter: brightness(1.3); }}
  .node-label {{ fill:#FAFAFA; font-size:12px; text-anchor:middle; pointer-events:none; }}
  .node-sublabel {{ fill:#888; font-size:10px; text-anchor:middle; pointer-events:none; }}
  .edge-line {{ stroke:#444; stroke-width:2px; marker-end:url(#arrowhead); }}
  .edge-line.active {{ stroke:#00FFAA; stroke-width:3px; stroke-dasharray:none; }}
  .stage-badge {{ fill:rgba(0,255,170,0.12); rx:8; }}
</style>
</head>
<body>
<svg id="dag" viewBox="0 0 760 480">
  <defs>
    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#444" id="arrow-fill"/>
    </marker>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <!-- Edges (drawn first, behind nodes) -->
  <!-- Stage 0 → 1 -->
  <line class="edge-line" id="e01" x1="76"  y1="240" x2="194" y2="240"/>
  <!-- Stage 1 → 2 -->
  <line class="edge-line" id="e12" x1="226" y1="240" x2="344" y2="240"/>
  <!-- Stage 2 → 3 -->
  <line class="edge-line" id="e23" x1="376" y1="240" x2="494" y2="240"/>
  <!-- Stage 3 → 4 -->
  <line class="edge-line" id="e34" x1="526" y1="240" x2="644" y2="240"/>

  <!-- Stage nodes -->
  <!-- 0: Data Ingestion -->
  <g id="node0" onclick="setStage(0)">
    <circle class="node-circle" cx="38"  cy="240" r="34" fill="#1E222B" stroke="#3399FF"/>
    <text class="node-label"    x="38"   y="235">DATA</text>
    <text class="node-sublabel" x="38"   y="250">INGEST</text>
  </g>

  <!-- 1: Target Formulation -->
  <g id="node1" onclick="setStage(1)">
    <circle class="node-circle" cx="210" cy="240" r="34" fill="#1E222B" stroke="#3399FF"/>
    <text class="node-label"    x="210"  y="235">TARGET</text>
    <text class="node-sublabel" x="210"  y="250">FORM.</text>
  </g>

  <!-- 2: Representation Learning -->
  <g id="node2" onclick="setStage(2)">
    <circle class="node-circle" cx="380" cy="240" r="34" fill="#1E222B" stroke="#3399FF"/>
    <text class="node-label"    x="380"  y="231">REPR.</text>
    <text class="node-sublabel" x="380"  y="245">LEARN</text>
    <text class="node-sublabel" x="380"  y="258">(NLP+GNN)</text>
  </g>

  <!-- 3: Multimodal Fusion -->
  <g id="node3" onclick="setStage(3)">
    <circle class="node-circle" cx="540" cy="240" r="34" fill="#1E222B" stroke="#3399FF"/>
    <text class="node-label"    x="540"  y="235">FUSION</text>
    <text class="node-sublabel" x="540"  y="250">248-dim</text>
  </g>

  <!-- 4: Inference -->
  <g id="node4" onclick="setStage(4)">
    <circle class="node-circle" cx="710" cy="240" r="34" fill="#1E222B" stroke="#3399FF"/>
    <text class="node-label"    x="710"  y="235">INFER</text>
    <text class="node-sublabel" x="710"  y="250">+ SHAP</text>
  </g>

  <!-- Sub-labels beneath each node (data source annotations) -->
  <text x="38"  y="290" text-anchor="middle" fill="#555" font-size="9" font-family="Courier New">Bloomberg</text>
  <text x="38"  y="301" text-anchor="middle" fill="#555" font-size="9" font-family="Courier New">+ EDGAR</text>

  <text x="210" y="290" text-anchor="middle" fill="#555" font-size="9" font-family="Courier New">CAR [-5,+5]</text>
  <text x="210" y="301" text-anchor="middle" fill="#555" font-size="9" font-family="Courier New">Market Model</text>

  <text x="380" y="290" text-anchor="middle" fill="#555" font-size="9" font-family="Courier New">FinBERT</text>
  <text x="380" y="301" text-anchor="middle" fill="#555" font-size="9" font-family="Courier New">HeteroSAGE</text>

  <text x="540" y="290" text-anchor="middle" fill="#555" font-size="9" font-family="Courier New">XGBoost</text>
  <text x="540" y="301" text-anchor="middle" fill="#555" font-size="9" font-family="Courier New">5-fold CV</text>

  <text x="710" y="290" text-anchor="middle" fill="#555" font-size="9" font-family="Courier New">SHAP</text>
  <text x="710" y="301" text-anchor="middle" fill="#555" font-size="9" font-family="Courier New">H1/H2/H3</text>

  <!-- AUC annotation bar at bottom -->
  <rect x="20" y="340" width="720" height="1" fill="#333"/>
  <text x="380" y="360" text-anchor="middle" fill="#555" font-size="10" font-family="Courier New">
    M1 AUC 0.541  →  M2 AUC 0.529  →  M3 AUC 0.566 ✅  (n=2,864 · 5-fold CV)
  </text>
</svg>

<script>
  const ACTIVE_STAGE = {stage_idx};

  const COLORS = {{
    inactive: "#3399FF",
    active:   "#00FFAA",
    edge_inactive: "#444",
    edge_active:   "#00FFAA",
  }};

  function setStage(idx) {{
    // Highlight nodes
    for (let i = 0; i < 5; i++) {{
      const circle = document.querySelector("#node" + i + " circle");
      const labels = document.querySelectorAll("#node" + i + " text");
      if (i === idx) {{
        circle.setAttribute("stroke", COLORS.active);
        circle.setAttribute("fill",   "rgba(0,255,170,0.12)");
        circle.style.filter = "url(#glow)";
        labels.forEach(l => l.setAttribute("fill", "#00FFAA"));
      }} else {{
        circle.setAttribute("stroke", COLORS.inactive);
        circle.setAttribute("fill",   "#1E222B");
        circle.style.filter = "";
        labels.forEach(l => {{
          if (l.classList.contains("node-label"))    l.setAttribute("fill", "#FAFAFA");
          if (l.classList.contains("node-sublabel")) l.setAttribute("fill", "#888");
        }});
      }}
    }}

    // Highlight edges leading INTO the active node
    const edgeMap = {{ 1:"e01", 2:"e12", 3:"e23", 4:"e34" }};
    ["e01","e12","e23","e34"].forEach(id => {{
      const el = document.getElementById(id);
      el.setAttribute("stroke", COLORS.edge_inactive);
      el.setAttribute("stroke-width", "2");
      el.setAttribute("stroke-dasharray", "");
    }});
    if (edgeMap[idx]) {{
      const el = document.getElementById(edgeMap[idx]);
      el.setAttribute("stroke", COLORS.edge_active);
      el.setAttribute("stroke-width", "3");
      // Animate dash
      el.setAttribute("stroke-dasharray", "200");
      el.setAttribute("stroke-dashoffset", "200");
      el.style.transition = "stroke-dashoffset 0.6s ease";
      setTimeout(() => {{ el.style.strokeDashoffset = "0"; }}, 10);
    }}

    // Update arrow fill colour
    document.getElementById("arrow-fill").setAttribute("fill",
      idx > 0 ? COLORS.edge_active : COLORS.edge_inactive);
  }}

  // Run on load with the injected stage index
  window.addEventListener("DOMContentLoaded", () => setStage(ACTIVE_STAGE));
</script>
</body>
</html>"""

    components.html(dag_html, height=380)

    # ── Stage detail expanders ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📂 View Source Code", expanded=False):
        sources = STAGE_META[stage_idx]["sources"]
        st.markdown("**Scripts executed in this stage:**")
        for src in sources:
            colour = "#FFB300" if any(k in src for k in ["retry", "fix_dates"]) else "#00FFAA"
            st.markdown(
                f"<code style='color:{colour};'>{src}</code>",
                unsafe_allow_html=True,
            )

    with st.expander("📊 Stage Output Statistics", expanded=True):
        stats = STAGE_META[stage_idx]["stats"]
        rows = ""
        for k, v in stats.items():
            v_col = "#00FFAA" if "✅" in str(v) else "#FAFAFA"
            rows += (
                f"<tr>"
                f"<td style='padding:6px 12px 6px 0; color:#888; font-family:monospace;'>{k}</td>"
                f"<td style='padding:6px 0; color:{v_col}; font-family:monospace; font-weight:bold;'>{v}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<table style='width:100%; border-collapse:collapse;'>{rows}</table>",
            unsafe_allow_html=True,
        )
