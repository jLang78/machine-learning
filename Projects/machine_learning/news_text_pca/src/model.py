
import os
import sys
import pandas as pd
import joblib
from sklearn.decomposition import PCA

# --- find path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ------------
def run_pca(n_components=2, processed_dir="data/processed", model_dir="models"):
    """
    Loads the TF-IDF matrix, applies PCA, and appends the 2D coordinates
    to the processed CSV for easy visualsation.
    """
    # 1. Setup file paths
    matrix_path = os.path.join(project_root, processed_dir, "tfidf_matrix.pkl")
    df_path = os.path.join(project_root, processed_dir, "processed_news.csv")

    print("Loading TF-IDF matrix and processed text data...")
    tfidf_matrix = joblib.load(matrix_path)
    df = pd.read_csv(df_path)

    # 2. Initialise PCA
    # I must convert the sparse TF-IDF matrix to a dense array for standard PCA
    print(f"Applying PCA to reduce {tfidf_matrix.shape[1]} dimensions down to {n_components}...")
    dense_matrix = tfidf_matrix.toarray()
    pca = PCA(n_components=n_components)

    # 3. Fit and transform the data
    pca_features = pca.fit_transform(dense_matrix)

    # 4. Analyse explained variance
    # This tells me how much "information" was retained after crushing the dimensions
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA completed")
    print(f"The {n_components} components explain {explained_variance:.2f}% of the total variance in the text.")

    # 5. Append the new mathematical coordinates to the original DataFrame
    df['pca_x'] = pca_features[:, 0]
    df['pca_y'] = pca_features[:, 1]

    # If later I decided to do 3D mapping, I add the Z axis
    if n_components >= 3:
        df['pca_z'] = pca_features[:, 2]

    # 6. Save the updated DataFrame and the PCA model
    df.to_csv(df_path, index=False)

    pca_model_path = os.path.join(project_root, model_dir, "pca_model.pkl")
    joblib.dump(pca, pca_model_path)

    print(f"Saved updated dataset with PCA coordinates to {df_path}")
    print(f"Saved fitted PCA model to {pca_model_path}")


if __name__ == "__main__":
    # I use 2 components so so I can easily plot it on a standard X/Y 2D graph!
    run_pca(n_components=2)