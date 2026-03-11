# src/features.py
import os
import sys
import pandas as pd

# --- PATH FIX ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ----------------

def process_and_scale_images(input_filename="fashion_mnist_raw.csv", output_dir="data/processed"):
    """
    Loads raw flattened images, normalises pixel values to 0-1, and saves for modeling.
    """
    raw_path = os.path.join(project_root, "data", "raw", input_filename)
    processed_dir = os.path.join(project_root, output_dir)
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Loading raw data from {raw_path}...")

    if not os.path.exists(raw_path):
        print(f"ERROR: Could not find {raw_path}. Did you run data.py?")
        return

    df = pd.read_csv(raw_path)

    print("Separating pixels from labels...")
    # grab all 784 column names that start with 'pixel'
    pixel_cols = [col for col in df.columns if col.startswith('pixel')]

    # Extract features (X) and metadata (y)
    X = df[pixel_cols]
    metadata = df[['label', 'category']]

    print("Normalising pixel values (scaling from 0-255 to 0.0-1.0)...")
    # Scaling helps distance-based algorithms like t-SNE converge faster and more accurately
    X_scaled = X / 255.0

    # Recombine the metadata and the newly scaled pixels to save
    df_processed = pd.concat([metadata, X_scaled], axis=1)

    output_path = os.path.join(processed_dir, "fashion_mnist_scaled.csv")
    print(f"Saving processed data to {output_path}...")

    # I save as CSV so I can easily inspect it if needed
    df_processed.to_csv(output_path, index=False)

    print(f"Success! Scaled data saved. Ready for t-SNE.")


if __name__ == "__main__":
    process_and_scale_images()