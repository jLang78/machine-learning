# src/model.py
import os
import sys
import pandas as pd
import joblib
from sklearn.manifold import TSNE

# --- Finding path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ----------------

def run_tsne(perplexity=30, n_components=2):
    """
    I load the scaled image data, apply t-SNE to reduce the 784 dimensions,
    and save the resulting coordinates for visualisation.
    """
    processed_dir = os.path.join(project_root, "data", "processed")
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)

    input_path = os.path.join(processed_dir, "fashion_mnist_scaled.csv")

    print(f"Loading scaled data from {input_path}...")
    df = pd.read_csv(input_path)

    # I isolate the pixel columns from the labels and categories
    pixel_cols = [col for col in df.columns if col.startswith('pixel')]
    X = df[pixel_cols]

    print(f"Applying t-SNE with perplexity={perplexity}. This takes a moment...")

    # I initialise the t-SNE algorithm
    # Perplexity controls how t-SNE balances local and global aspects of the data
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)

    # I fit the model and transform the data into 2D coordinates
    # Unlike PCA, t-SNE does not have a separate transform method for new data
    tsne_results = tsne.fit_transform(X)

    print("t-SNE complete. Saving coordinates...")

    # I attach the new X and Y coordinates back to the original dataframe
    df['tsne_x'] = tsne_results[:, 0]
    df['tsne_y'] = tsne_results[:, 1]

    # I save the updated dataframe so the visualisation script can read it
    output_csv = os.path.join(processed_dir, "fashion_mnist_tsne.csv")
    df.to_csv(output_csv, index=False)

    # I save the raw t-SNE array as a pickle file
    output_pkl = os.path.join(model_dir, "tsne_coordinates.pkl")
    joblib.dump(tsne_results, output_pkl)

    print(f"Saved dataset with t-SNE coordinates to {output_csv}")
    print(f"Saved raw coordinates to {output_pkl}")


if __name__ == "__main__":
    run_tsne(perplexity=30)