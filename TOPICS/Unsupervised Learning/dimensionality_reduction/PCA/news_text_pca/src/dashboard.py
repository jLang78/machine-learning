import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

# --- PATH FIX ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ----------------

# Set up the page layout
st.set_page_config(page_title="PCA News Clustering", layout="wide")

st.title("News Topic Clustering with PCA")
st.markdown("""
This dashboard demonstrates **Dimensionality Reduction** using Principal Component Analysis (PCA).
I fetched live news data, transformed the text into a TF-IDF mathematical matrix, and reduced 
thousands of dimensions down to just two coordinates (X and Y) to visualise how different topics group together.
""")


# 1. Load the data
@st.cache_data  # This tells Streamlit to cache the data so it doesn't reload on every button click
def load_data():
    df_path = os.path.join(project_root, "data", "processed", "processed_news.csv")
    if os.path.exists(df_path):
        return pd.read_csv(df_path)
    return None


df = load_data()

if df is not None:
    # Format text for hover
    df['hover_text'] = df['raw_text'].apply(lambda x: x[:120] + "..." if len(str(x)) > 120 else x)

    # Sidebar for interactivity
    st.sidebar.header("Dashboard Controls")
    categories = df['category'].unique().tolist()
    selected_categories = st.sidebar.multiselect(
        "Select Topics to Display:",
        options=categories,
        default=categories
    )

    # Filter data based on user selection
    filtered_df = df[df['category'].isin(selected_categories)]

    # 2. Build the Plotly Graph
    fig = px.scatter(
        filtered_df,
        x="pca_x",
        y="pca_y",
        color="category",
        hover_name="category",
        hover_data={"hover_text": True, "pca_x": False, "pca_y": False, "category": False},
        template="plotly_dark",
        height=600
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))

    # 3. Render the graph in the web app
    st.plotly_chart(fig, use_container_width=True)

    # 4. Show the raw data below the graph
    with st.expander("View Raw Processed Data"):
        st.dataframe(filtered_df[['category', 'raw_text', 'pca_x', 'pca_y']])

else:
    st.error("No processed data found. Please run features.py and model.py first!")