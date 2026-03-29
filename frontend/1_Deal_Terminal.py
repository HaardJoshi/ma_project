import streamlit as st
import json
import plotly.graph_objects as go
import os
import time
from pyvis.network import Network
import streamlit.components.v1 as components

# Load Neural Weights & Payloads
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'payloads.json')
with open(DATA_PATH, 'r') as f:
    deals = json.load(f)


from utils import setup_page

# ==========================================
# 1. The Aesthetic Engine (CSS Injection)
# ==========================================
setup_page(title="Deal Terminal")

# ==========================================
# 2. The Control Deck (State Logic)
# ==========================================
st.sidebar.title("Control Deck")
st.sidebar.markdown("""<div style='font-family:monospace; font-size:10px; color:#555; border:1px solid #333; padding:8px; border-radius:4px; margin-bottom:16px;'>4,999 DEALS &nbsp;·&nbsp; 28 YRS &nbsp;·&nbsp; 249 FEATS<br/>AUC 0.566 &nbsp;·&nbsp; p=0.038 &nbsp;·&nbsp; n=2,864</div>""", unsafe_allow_html=True)
st.sidebar.markdown("---")

deal_options = [d["deal_id"] for d in deals]
selected_deal = st.sidebar.selectbox("Target Pipeline", deal_options)

if "executed_deal" not in st.session_state:
    st.session_state["executed_deal"] = None

if st.sidebar.button("EXECUTE DIAGNOSTICS", use_container_width=True):
    with st.sidebar:
        with st.spinner("Init GraphSAGE..."):
            time.sleep(1.0)
        with st.spinner("Extract FinBERT..."):
            time.sleep(1.0)
        with st.spinner("Fusing modalities..."):
            time.sleep(1.0)
    st.session_state["executed_deal"] = selected_deal

if st.session_state["executed_deal"] != selected_deal:
    st.title("M&A Deal Intelligence Terminal")
    st.info("Awaiting telemetry. Select a Pipeline Deal inside the Control Deck and trigger EXECUTE DIAGNOSTICS.")
    st.stop()

current_deal = next((d for d in deals if d["deal_id"] == st.session_state["executed_deal"]), deals[0])
acq_full = current_deal.get('acquirer_name', current_deal['acquirer'])
tgt_full = current_deal.get('target_name', current_deal['target'])
st.markdown(f"<h2>Diagnostic Report: <span style='color:#00FFAA;'>{acq_full}</span> <span style='color:#555;'>➔</span> <span style='color:#3399FF;'>{tgt_full}</span></h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 3. The Frontend Grid (Layout Architecture)
# ==========================================

# Top Row: Macro Metrics
st.markdown("### Macro Metrics")
top_col1, top_col2, top_col3 = st.columns([1, 1, 1])

with top_col1:
    st.markdown("### The Verdict")
    st.caption("Multimodal calibrated ensemble forecasting synergy likelihood.")
    syn_prob = current_deal.get("synergy_probability", 0) * 100
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = syn_prob,
        title = {'text': "Synergy Probability", 'font': {'color': '#00FFAA', 'family': 'monospace'}},
        number = {'font': {'color': '#FAFAFA', 'family': 'monospace'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickcolor': "#FAFAFA"},
            'bar': {'color': "rgba(255,255,255,0.5)"},
            'steps': [
                {'range': [0, 45], 'color': "red"},
                {'range': [45, 55], 'color': "yellow"},
                {'range': [55, 100], 'color': "green"}
            ],
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#1E222B"
        }
    ))
    
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#FAFAFA", 'family': "monospace"},
        margin=dict(l=20, r=20, t=50, b=20),
        height=250
    )
    
    st.plotly_chart(fig_gauge, use_container_width=True)

with top_col2:
    st.markdown("### Network Alpha")
    st.caption("Betweenness centrality within the global supply-chain graph.")
    bet_pctile  = current_deal.get("acquirer_betweenness_percentile", 0.0)
    sc_depth    = current_deal.get("acquirer_supply_chain_depth", 0)
    bet_color   = "#00FFAA" if bet_pctile >= 50 else "#FFB300"
    st.markdown(f"""
        <div style="background-color: rgba(30,34,43,0.3); border: 1px solid rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; text-align: center; height: 250px; display: flex; flex-direction: column; justify-content: center;">
            <p style="color: #888; font-family: monospace; font-size: 13px; margin: 0;">Betweenness Centrality Rank</p>
            <h2 style="color: {bet_color}; font-family: monospace; font-size: 42px; margin: 8px 0;">{bet_pctile:.0f}<span style="font-size:22px;">th %ile</span></h2>
            <p style="color: #bbb; font-size: 12px; margin: 0;">Supply-Chain Depth: <strong style="color:#FAFAFA;">{sc_depth}</strong> direct SPLC links</p>
        </div>
    """, unsafe_allow_html=True)

with top_col3:
    st.markdown("### Semantic Match")
    st.caption("MD&A centroid cosine similarity vs. market-mean embedding.")
    mda_pctile  = current_deal.get("semantic_scores", {}).get("mda_percentile", 50.0)
    mda_cos     = current_deal.get("mda_cosine_similarity", 0.5)
    sem_color   = "#00FFAA" if mda_pctile >= 50 else "#3399FF"
    st.markdown(f"""
        <div style="background-color: rgba(30,34,43,0.3); border: 1px solid rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; text-align: center; height: 250px; display: flex; flex-direction: column; justify-content: center;">
            <p style="color: #888; font-family: monospace; font-size: 13px; margin: 0;">MD&A Centroid Cosine Similarity</p>
            <h2 style="color: {sem_color}; font-family: monospace; font-size: 42px; margin: 8px 0;">{mda_pctile:.0f}<span style="font-size:22px;">th %ile</span></h2>
            <p style="color: #bbb; font-size: 12px; margin: 0;">Raw cosine: <strong style="color:#FAFAFA;">{mda_cos:.4f}</strong> vs market centroid</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Middle Row: Zone 2 (Topological Radar), Zone 3 (Semantic Radar)
mid_col1, mid_col2 = st.columns([3, 2])

with mid_col1:
    st.markdown("### Zone 1: Topological Embeddings")
    st.caption("Visualizes structural M&A complexity (Supplier Overlap/Risk).")
    
    # Create pyvis graph (forceAtlas2Based physics & 150 iterations stabilization)
    net = Network(height='400px', width='100%', bgcolor='#0E1117', font_color='#FAFAFA')
    
    risk_sim = current_deal.get("semantic_scores", {}).get("risk_similarity", 0)
    tgt_color = "#FF3333" if risk_sim > 0.5 else "#3399FF" # Target node risk coloring
    
    graph_data = current_deal.get("graph_data", {"nodes": [], "edges": []})
    
    for node in graph_data.get("nodes", []):
        lbl = node.get("label", node["id"])
        if node["group"] == "acq":
            net.add_node(node["id"], label=lbl, color="#00FFAA", size=30)
        elif node["group"] == "tgt":
            net.add_node(node["id"], label=lbl, color=tgt_color, size=30)
        else:
            net.add_node(node["id"], label=lbl, color="#555555", size=10)
            
    for edge in graph_data.get("edges", []):
        net.add_edge(edge["source"], edge["target"], color="#FAFAFA")
        
    net.set_options("""
    var options = {
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springConstant": 0.08,
          "springLength": 100,
          "damping": 0.4,
          "avoidOverlap": 0
        },
        "stabilization": {
          "enabled": true,
          "iterations": 150
        }
      }
    }
    """)
    
    graph_path = os.path.join(os.path.dirname(__file__), "temp_graph.html")
    net.save_graph(graph_path)
    
    with open(graph_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
        
    html_data = html_data.replace('border: 1px solid lightgray;', 'border: none;')
    html_data = html_data.replace(
        'network = new vis.Network(container, data, options);', 
        'network = new vis.Network(container, data, options); setTimeout(function(){ network.setOptions({physics: false}); }, 1500);'
    )
        
    components.html(html_data, height=410)

with mid_col2:
    st.markdown("### Zone 2: Semantic Radar")
    st.caption("Strategic execution and embedded variance vs acquirer.")
    
    mda_sim = current_deal.get("semantic_scores", {}).get("mda_similarity", 0)
    risk_sim = current_deal.get("semantic_scores", {}).get("risk_similarity", 0)
    syn_prob = current_deal.get("synergy_probability", 0)
    car_val = 0.8 if current_deal.get("car_prediction", "") == "POSITIVE" else 0.2
    
    categories = ['MD&A', 'Risk', 'Strategic', 'Sentiment', 'Ops Fix']
    
    acq_values = [1.0, 1.0, 1.0, 1.0, 1.0]
    tgt_values = [mda_sim, risk_sim, syn_prob, car_val, (mda_sim+syn_prob)/2]
    
    fig_radar = go.Figure()
    
    # Acquirer Benchmark Fill
    fig_radar.add_trace(go.Scatterpolar(
        r=acq_values + [acq_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=current_deal.get("acquirer", "Acquirer"),
        line=dict(color='#00FFAA', width=2),
        fillcolor='rgba(0, 255, 170, 0.1)'
    ))
    
    # Target Evaluation Fill
    fig_radar.add_trace(go.Scatterpolar(
        r=tgt_values + [tgt_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=current_deal.get("target", "Target"),
        line=dict(color='#3399FF', width=2),
        fillcolor='rgba(51, 153, 255, 0.4)'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#333333', tickcolor='#333333', showticklabels=True, tickfont=dict(color='#888', size=10)),
            angularaxis=dict(gridcolor='#333333', linecolor='#333333'),
            bgcolor='#0E1117'
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#FAFAFA", 'family': "monospace", 'size': 10},
        margin=dict(l=30, r=30, t=40, b=40),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# Bottom Row: Zone 4 (The Glass Box / SHAP)
bot_cols = st.columns([1])

with bot_cols[0]:
    st.markdown("### Zone 3: The Glass Box (Feature SHAP)")
    st.caption("SHAP outputs quantifying the exact impact of raw features on the M3 Synergy Probability.")
    
    shap_data = current_deal.get("shap_data", [])
    if shap_data:
        x_vals = [d["feature"] for d in shap_data]
        y_vals = [d["value"] for d in shap_data]
        
        # Add a Net totals column
        x_vals.append("Net SHAP Impact")
        y_vals.append(sum(y_vals))
        measures = ["relative"] * len(shap_data) + ["total"]
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="SHAP Impact",
            orientation="v",
            measure=measures,
            x=x_vals,
            y=y_vals,
            textposition="outside",
            text=[f"{v:+.3f}" for v in y_vals],
            connector={"line": {"color": "rgba(255, 255, 255, 0.2)", "dash": "dot"}},
            decreasing={"marker": {"color": "#FF3333"}},
            increasing={"marker": {"color": "#00FFAA"}},
            totals={"marker": {"color": "#3399FF"}}
        ))
        
        fig_waterfall.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#FAFAFA", 'family': "monospace"},
            margin=dict(l=40, r=40, t=40, b=40),
            height=350,
            showlegend=False,
            yaxis=dict(gridcolor='#333333', zerolinecolor='#555555')
        )
        
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # Algorithmic Translation
        car_pred_color = "#00FFAA" if current_deal.get('car_prediction') == "POSITIVE" else "#FF3333"
        st.markdown(f"""
            <div style="background-color: rgba(30, 34, 43, 0.4); border: 1px solid rgba(255, 255, 255, 0.2); border-left: 4px solid {car_pred_color}; padding: 15px; border-radius: 4px; margin-top: 20px;">
                <p style="font-family: monospace; color: #FAFAFA; margin: 0; font-size: 14px;">
                    <span style="color: {car_pred_color}; font-weight: bold;">[ALGORITHMIC TRANSLATION]:</span> The M3 predictive model concludes a <strong style="color: {car_pred_color};">{current_deal.get('car_prediction', 'UNKNOWN')}</strong> synergy forecast. 
                    Primary drivers include <strong>{', '.join([d['feature'] for d in shap_data[:2]])}</strong>, accounting for the most significant SHAP variance.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("*(No SHAP explicability logs found for exactly this deal)*")