# src/dashboard.py
import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

# --- Finding path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ----------------

# I configure the Streamlit page layout and title
st.set_page_config(page_title="t-SNE Image Clustering", layout="wide")

st.title("Image Feature Clustering with t-SNE")
st.markdown("""
This dashboard visualises **Non-Linear Dimensionality Reduction**.
I took 10,000 images from the Fashion-MNIST dataset, flattened them into 784 dimensions, 
and used t-SNE to map their visual similarities in a 2D coordinate space.
""")


@st.cache_data
def load_tsne_data():
    """
    I load the processed dataset containing the final t-SNE coordinates.
    I use caching so the data does not reload on every UI interaction.
    """
    input_path = os.path.join(project_root, "data", "processed", "fashion_mnist_tsne.csv")
    if os.path.exists(input_path):
        return pd.read_csv(input_path)
    return None


df = load_tsne_data()

if df is not None:
    # I set up a sidebar for user filtering
    st.sidebar.header("Filter Clusters")
    categories = sorted(df['category'].unique().tolist())

    selected_categories = st.sidebar.multiselect(
        "Select Clothing Categories:",
        options=categories,
        default=categories
    )

    # I filter the dataframe based on the user's selection
    filtered_df = df[df['category'].isin(selected_categories)]

    # I build the interactive Plotly scatter plot
    fig = px.scatter(
        filtered_df,
        x="tsne_x",
        y="tsne_y",
        color="category",
        hover_name="category",
        template="plotly_dark",
        height=700,
        opacity=0.8
    )

    # I adjust marker size for readability
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))

    # I render the plot on the main screen
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Data not found. I need to run features.py and model.py first.")