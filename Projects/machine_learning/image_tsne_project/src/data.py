# src/data.py
# src/data.py
import os
import sys
import gzip
import struct
import requests
import numpy as np
import pandas as pd

# --- Finding path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ---------
def _download_file(url, dest_path):
    """Download a file using requests (which bundles its own CA certs)."""
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def _parse_idx_images(filepath):
    """Parse IDX image file format used by Fashion-MNIST."""
    with gzip.open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
    return data


def _parse_idx_labels(filepath):
    """Parse IDX label file format used by Fashion-MNIST."""
    with gzip.open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def download_fashion_mnist(output_dir="data/raw"):
    """
    Downloading the Fashion-MNIST dataset and saving locally.
    Uses direct HTTP download to avoid macOS SSL issues with fetch_openml.
    """
    raw_dir = os.path.join(project_root, output_dir)
    os.makedirs(raw_dir, exist_ok=True)

    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    print("Downloading Fashion-MNIST dataset (this might take a minute)...")

    local_paths = {}
    for key, filename in files.items():
        dest = os.path.join(raw_dir, filename)
        if not os.path.exists(dest):
            print(f"  Downloading {filename}...")
            _download_file(base_url + filename, dest)
        local_paths[key] = dest

    # Parse the gzip files
    train_images = _parse_idx_images(local_paths["train_images"])
    train_labels = _parse_idx_labels(local_paths["train_labels"])
    test_images = _parse_idx_images(local_paths["test_images"])
    test_labels = _parse_idx_labels(local_paths["test_labels"])

    # Combine train and test sets (70,000 total, same as fetch_openml)
    X = np.concatenate([train_images, test_images], axis=0)
    y = np.concatenate([train_labels, test_labels], axis=0)

    # Build a DataFrame with pixel columns named like the original
    pixel_columns = [f"pixel{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=pixel_columns)
    df['label'] = y.astype(str)

    # Map the numeric labels (0-9) to actual clothing names for the graphs
    label_map = {
        '0': 'T-shirt/top', '1': 'Trouser', '2': 'Pullover', '3': 'Dress', '4': 'Coat',
        '5': 'Sandal', '6': 'Shirt', '7': 'Sneaker', '8': 'Bag', '9': 'Ankle boot'
    }
    df['category'] = df['label'].map(label_map)

    # To keep the project running fast on a local machine, I sample 10,000 random images
    # instead of the full 70,000. t-SNE is computationally heavy, so this is a good tradeoff
    df_sampled = df.sample(n=10000, random_state=42).reset_index(drop=True)

    # Save the raw data
    output_path = os.path.join(raw_dir, "fashion_mnist_raw.csv")
    df_sampled.to_csv(output_path, index=False)

    print(f"Success! Saved {len(df_sampled)} images (as flattened pixel rows) to {output_path}")


if __name__ == "__main__":
    download_fashion_mnist()
