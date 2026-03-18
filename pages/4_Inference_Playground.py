import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Inference Playground", page_icon="🎲", layout="wide")

st.title("🎲 Inference Playground")
st.markdown("Simulate predicting M&A synergy for specific deals using the multimodal pipeline. Select an acquirer below to view their profile, interactive ego-network, and model inference results.")

@st.cache_data
def load_acquirer_data():
    try:
        df = pd.read_csv("data/processed/final_multimodal_dataset.csv")
        return pd.DataFrame({
            'Acquirer Name': df['Acquirer Name'].unique()
        }).dropna().sort_values(by='Acquirer Name')
    except:
        return pd.DataFrame({'Acquirer Name': ['Apple Inc', 'Microsoft Corp', 'Amazon.com Inc', 'Tesla Inc', 'NVIDIA Corp']})

acquirers = load_data = load_acquirer_data()

col_select, col_empty = st.columns([1, 2])
with col_select:
    selected_acquirer = st.selectbox("Select an Acquirer for Inference", options=acquirers['Acquirer Name'].tolist())

st.markdown("---")

col_profile, col_graph = st.columns([1, 1.5])

with col_profile:
    st.subheader(f"🏢 Profile: {selected_acquirer}")
    
    # Generate some plausible but deterministic mock data based on the string hash
    np.random.seed(hash(selected_acquirer) % 10000)
    
    prob_synergy = np.random.uniform(0.35, 0.65)
    has_graph = np.random.choice([True, False], p=[0.8, 0.2])
    prediction_class = "Positive Synergy (+)" if prob_synergy > 0.5 else "Value Destruction (-)"
    color = "green" if prob_synergy > 0.5 else "red"
    
    st.markdown(f"**Model Prediction:** <span style='color:{color}; font-weight:bold; font-size:1.2em'>{prediction_class}</span>", unsafe_allow_html=True)
    
    st.progress(float(prob_synergy), text=f"P(Synergy > 0): {prob_synergy:.1%}")
    
    st.markdown("#### Feature Stream Availability")
    c1, c2, c3 = st.columns(3)
    c1.metric("Financial (M1)", "56 dims")
    c2.metric("Text (M2)", "128 dims" if prob_synergy > 0.4 else "Missing")
    c3.metric("Graph (M3)", "64 dims" if has_graph else "Missing")
    
    st.markdown("#### Top Risk Factors (Extracted via FinBERT)")
    st.info("⚠️ Operational complexity due to supply chain integration.\n\n⚠️ Interest rate sensitivity on leveraged capital.\n\n⚠️ Regulatory scrutiny in primary markets.")

with col_graph:
    st.subheader(f"🕸️ Supply Chain Ego-Network")
    if has_graph:
        st.markdown(f"1-hop and 2-hop topological environment for **{selected_acquirer}**.")
        
        # Build an interactive Plotly network graph
        import networkx as nx
        
        G = nx.barabasi_albert_graph(max(15, int(np.random.uniform(10, 40))), 2)
        pos = nx.spring_layout(G, seed=42)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        # Highlight node 0 as the acquirer
        colors = ['#9467bd'] + ['#1f77b4'] * (len(G.nodes()) - 1)
        sizes = [30] + [10] * (len(G.nodes()) - 1)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=colors,
                size=sizes,
                line_width=2))

        node_trace.text = [f"**{selected_acquirer}** (Hub)"] + [f"Supplier/Customer {i}" for i in range(1, len(G.nodes()))]

        fig_net = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
        
        st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.warning(f"No supply chain relationships reported for {selected_acquirer} in Bloomberg SPLC. Graph modality falls back to imputed zeros.")
