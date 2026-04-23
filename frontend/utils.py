import streamlit as st
import json
import os

def load_macro_stats():
    """Loads the precomputed macro statistics json."""
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'macro_stats.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def enforce_bloomberg_css():
    """Injects the globally enforced Bloomberg Terminal dark mode CSS."""
    st.markdown(
        """<style>
        /* General Layout */
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; } 
        header {visibility: hidden;} 
        footer {visibility: hidden;} 
        
        /* Font and color styling globally */
        body { font-family: "Courier New", Courier, monospace !important; background-color: #0E1117;}
        div[data-testid="stMetricValue"] { font-family: "Courier New", Courier, monospace; color: #00FFAA; }
        
        /* Expanded State Navbar */
        section[data-testid="stSidebar"][aria-expanded="true"] { 
            width: 275px !important; min-width: 200px !important; 
        }
        
        /* Collapsed State Navbar: Reduced to 75px and 20% opacity background */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            transform: translateX(0px) !important;
            width: 75px !important;
            min-width: 75px !important;
            background-color: rgba(30, 34, 43, 0.2) !important;
            border-right: none !important;
        }
        
        /* Toned-down sidebar and toggle buttons perpetually */
        section[data-testid="stSidebar"] button, button[data-testid="collapsedControl"] {
            background-color: transparent !important;
            border: 1px solid rgba(0, 255, 170, 0.3) !important;
            border-radius: 6px !important;
            transition: all 0.2s ease-in-out !important;
            color: rgba(0, 255, 170, 0.8) !important;
        }
        
        section[data-testid="stSidebar"] button:hover, button[data-testid="collapsedControl"]:hover {
            background-color: rgba(0, 255, 170, 0.1) !important;
            border: 1px solid rgba(0, 255, 170, 0.7) !important;
            color: #00FFAA !important;
        }
        
        /* Hide the form components while collapsed to prevent bleed-through */
        section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarUserContent"] {
            opacity: 0 !important;
            pointer-events: none !important;
        }
        
        /* Radio button timeline styles */
        div.row-widget.stRadio > div {
            flex-direction: row;
            align-items: center;
        }
        </style>""", 
        unsafe_allow_html=True
    )

def setup_page(title="M&A Deal Intelligence"):
    st.set_page_config(
        layout="wide", 
        page_title=title, 
        initial_sidebar_state="expanded"
    )
    enforce_bloomberg_css()

def load_betweenness_data():
    """Reads betweenness_cache.json, sorts, computes true percentiles, and returns the top 12 nodes for D3."""
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'betweenness_cache.json')
    if os.path.exists(path):
        import json
        import numpy as np
        from scipy import stats
        with open(path, 'r') as f:
            data = json.load(f)
        
        vals = np.array([float(x) for x in data.values()])
        non_zero_vals = vals[vals > 0]
        
        nodes = []
        for k, v in data.items():
            val = float(v)
            if val > 0:
                pct = stats.percentileofscore(non_zero_vals, val, kind='strict')
            else:
                pct = 0.0
            nodes.append({'id': str(k), 'val': val, 'percentile': round(pct, 1)})
            
        sorted_nodes = sorted(nodes, key=lambda x: x['val'], reverse=True)
        return sorted_nodes[:12]
    return []
