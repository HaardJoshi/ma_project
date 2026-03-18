import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dataset Explorer", page_icon="📊", layout="wide")

st.title("📊 Dataset Explorer")
st.markdown("Explore the 4,999 M&A deals in the dataset used for synergy prediction. This page allows you to filter deals by year and sector, visualize distributions, and inspect individual deal records.")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/processed/final_multimodal_dataset.csv")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
        
    if 'Announce Date' in df.columns:
        df['Announce Date'] = pd.to_datetime(df['Announce Date'], errors='coerce')
        df['Year'] = df['Announce Date'].dt.year
    else:
        df['Year'] = 2000
        
    df['Current Acquirer SIC Code'] = df.get('Current Acquirer SIC Code', pd.Series(dtype=str)).fillna(0).astype(int).astype(str)
    
    def get_sector(sic_str):
        try:
            sic = int(sic_str)
            if 2000 <= sic <= 3999: return "Manufacturing"
            elif 4000 <= sic <= 4999: return "Transport & Utilities"
            elif 5000 <= sic <= 5999: return "Wholesale & Retail"
            elif 6000 <= sic <= 6799: return "Finance & Real Estate"
            elif 7000 <= sic <= 8999: return "Services"
            else: return "Other"
        except:
            return "Other"

    df['Sector'] = df['Current Acquirer SIC Code'].apply(get_sector)
    return df

with st.spinner("Loading dataset..."):
    df = load_data()

if df.empty:
    st.stop()

# Basic filters
st.sidebar.header("Filter Dataset")
min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
selected_years = st.sidebar.slider("Select Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

sectors = sorted(df['Sector'].unique())
selected_sectors = st.sidebar.multiselect("Select Acquirer Sectors", options=sectors, default=sectors)

filtered_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sectors)]

# Top metrics
st.markdown("### Executive Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Deals", f"{len(filtered_df):,}")

if 'car_m5_p5' in filtered_df.columns:
    avg_car = filtered_df['car_m5_p5'].mean() * 100
    pos_car_pct = (filtered_df['car_m5_p5'] > 0).mean() * 100
    col2.metric("Average CAR", f"{avg_car:.2f}%")
    col3.metric("Positive Synergy", f"{pos_car_pct:.1f}%")
else:
    col2.metric("Average CAR", "N/A")
    col3.metric("Positive Synergy", "N/A")
    
has_graph_pct = filtered_df['has_graph'].mean() * 100 if 'has_graph' in filtered_df.columns else 0
col4.metric("Graph Coverage", f"{has_graph_pct:.1f}%")

st.markdown("---")
st.markdown("### Interactive Visualizations")

col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    yearly_counts = filtered_df.groupby('Year').size().reset_index(name='Count')
    fig1 = px.bar(yearly_counts, x='Year', y='Count', title="Number of Deals by Year", 
                  color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig1, use_container_width=True)

with col_viz2:
    if 'car_m5_p5' in filtered_df.columns:
        fig2 = px.box(filtered_df, x='Sector', y='car_m5_p5', title="CAR Distribution by Sector",
                      color='Sector')
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("### Financial Feature Explorer")
numeric_cols = [c for c in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[c]) and c not in ['Year', 'car_m5_p5', 'has_graph'] and 'Embedding' not in c and 'PCA' not in c and 'Unnamed' not in c]
numeric_cols = sorted(numeric_cols)

if numeric_cols:
    default_x = 'Announced Total Value (mil.)' if 'Announced Total Value (mil.)' in numeric_cols else numeric_cols[0]
    x_axis = st.selectbox("Select X-axis Feature for Scatter Plot", options=numeric_cols, index=numeric_cols.index(default_x) if default_x in numeric_cols else 0)
    
    if 'car_m5_p5' in filtered_df.columns and x_axis:
        fig3 = px.scatter(filtered_df, x=x_axis, y='car_m5_p5', color='Sector', 
                          hover_data=['Acquirer Name', 'Target Name', 'Announce Date'],
                          title=f"{x_axis} vs CAR (Cumulative Abnormal Return)", alpha=0.6)
        st.plotly_chart(fig3, use_container_width=True)

st.markdown("### Dataset Preview")
cols_to_show = ['Announce Date', 'Acquirer Name', 'Target Name', 'Sector', 'Announced Total Value (mil.)', 'car_m5_p5']
cols_to_show = [c for c in cols_to_show if c in filtered_df.columns]
st.dataframe(filtered_df[cols_to_show].head(100), use_container_width=True)
