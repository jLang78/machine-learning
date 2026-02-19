# src/visualise.py
import os
import sys
import pandas as pd
import plotly.express as px

# --- Finding path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ----------------

def plot_tsne_clusters():
    """
    I read the processed CSV containing the t-SNE coordinates and generate
    an interactive scatter plot using Plotly.
    """
    processed_dir = os.path.join(project_root, "data", "processed")
    input_path = os.path.join(processed_dir, "fashion_mnist_tsne.csv")

    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}. I need to run model.py first.")
        return

    print("Loading t-SNE coordinate data for visualisation...")
    df = pd.read_csv(input_path)

    print("Generating interactive Plotly graph...")
    # I plot the X and Y coordinates, coloring the points by their clothing category
    fig = px.scatter(
        df,
        x="tsne_x",
        y="tsne_y",
        color="category",
        hover_name="category",
        title="Fashion-MNIST: Visualising 784D Images in 2D with t-SNE",
        template="plotly_dark",
        opacity=0.7
    )

    # I adjust the marker size to prevent overlapping, given the high volume of data points
    fig.update_traces(marker=dict(size=4, line=dict(width=0.2, color='DarkSlateGrey')))

    # I open the graph in the default web browser
    fig.show()

    # I save a static HTML copy for my portfolio
    reports_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(reports_dir, exist_ok=True)

    html_path = os.path.join(reports_dir, "tsne_clusters.html")
    fig.write_html(html_path)
    print(f"Saved interactive HTML graph to {html_path}")


if __name__ == "__main__":
    plot_tsne_clusters()