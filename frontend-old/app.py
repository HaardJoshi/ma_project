import streamlit as st

st.set_page_config(
    page_title="M&A Synergy Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("M&A Synergy Explorer 📈")

st.markdown("""
Welcome to the **M&A Synergy Explorer**, an interactive dashboard accompanying the dissertation on M&A synergy prediction using multimodal machine learning.

### Dashboard Navigation
- **📊 Dataset Explorer**: Explore the 4,999 M&A deals, filtering by sector and year. Interactive scatter plots of financial features vs CAR and summary statistics.
- **🚀 Model Performance**: Review XGBoost classification metrics, ROC curves, and SHAP feature importance across the M1, M2, and M3 configurations.
- **🔬 Hypothesis Deep-Dive**: Interact with visualizations testing the three core hypotheses:
  - **H1:** Topological Alpha
  - **H2:** Semantic Divergence
  - **H3:** Topological Arbitrage
- **🎲 Inference Playground**: Select an individual deal to view its supply chain ego-network, multimodal feature summary, and model prediction with SHAP force plots.

Please use the sidebar to navigate to the different sections.
""")

st.sidebar.success("Select a page above.")

st.markdown("---")
st.markdown("Developed as part of the M&A synergy prediction dissertation methodology.")
