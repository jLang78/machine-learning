
import os
import sys
import pandas as pd
import plotly.express as px

# --- PATH FIX ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ----------------

def plot_pca_clusters(processed_dir="data/processed"):
    """
    Reads the processed CSV and generates an interactive Plotly scatter plot.
    """
    # 1. Load the data with our newly minted PCA coordinates
    df_path = os.path.join(project_root, processed_dir, "processed_news.csv")

    if not os.path.exists(df_path):
        print(f"ERROR: Could not find {df_path}. Did you run model.py yet?")
        return

    print("Loading data for visualisation...")
    df = pd.read_csv(df_path)

    # Wrap the raw text so it doesn't run off the screen in the tooltip
    df['hover_text'] = df['raw_text'].apply(lambda x: x[:100] + "..." if len(str(x)) > 100 else x)

    # 2. Generate the Plotly Scatter Plot
    print("Generating interactive Plotly graph...")
    fig = px.scatter(
        df,
        x="pca_x",
        y="pca_y",
        color="category",  # Colors the dots based on the news topic
        hover_name="category",  # Bold title in the hover box
        hover_data={"hover_text": True, "pca_x": False, "pca_y": False, "category": False},
        title="News Topics Clustered by TF-IDF & PCA",
        template="plotly_dark",  # dark mode theme
        opacity=0.8
    )

    # Make the dots larger
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))

    # 3. Open the graph in default web browser
    fig.show()

    # 4. Save a static HTML copy for your portfolio
    html_path = os.path.join(project_root, "reports", "figures", "pca_clusters.html")
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    fig.write_html(html_path)
    print(f"Saved interactive HTML graph to {html_path}")


if __name__ == "__main__":
    plot_pca_clusters()