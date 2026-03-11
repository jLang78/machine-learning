# Dimensionality Reduction: Image Feature Clustering with t-SNE

## Project Overview
In this project, I demonstrate the usefulness of Non-Linear Dimensionality Reduction by applying **t-SNE** (t-distributed Stochastic Neighbor Embedding) to computer vision data. 

I utilise the Fashion-MNIST dataset, processing 10,000 grayscale images of clothing. By flattening each 28x28 image into a 784-dimensional array and normalising the pixel values, I use t-SNE to mathematically map visual similarities into a 2D coordinate space without providing the algorithm any explicit label data.

## Business Value
Unsupervised clustering of image data, is, of course, of significant value in modern computer vision applications:
* **Visual Search & Recommendaton:** E-commerce platforms can recommend visually similar products - such as suggesting a visually similar sneaker if the current one is out of stock - by finding the nearest neighbors in the reduced coordinate space.
* **Automated Tagging:** Grouping thousands of untagged product images into logical categories to reduce manual data entry.
* **Dataset Quality Control:** Identifying mislabeled images by finding data points that map far outside their expected categorical cluster.

## Exploration & Analysis (Jupyter Notebooks)
Before constructing the automated pipeline, I used Jupyter Notebooks to explore the data and tune the hyperparameters:
* **`01_image_exploration.ipynb`**: I ingest the raw, flattened 784-dimensional vectors and successfully reconstruct them into 28x28 pixel grids using Matplotlib, confirming the integrity of the raw data.
* **`02_tsne_perplexity.ipynb`**: I conduct an empirical study of the `perplexity` hyperparameter. I demonstrate that low values (e.g., 5) fracture the data into meaningless micro-clusters, while excessively high values (e.g., 100) cause distinct categories to merge. I conclude that a perplexity of 30 provides the optimal balance between local and global data structure.

## Production Architecture
I structure the codebase as a robust, modular pipeline:
1. **`src/data.py`**: I execute a direct HTTP download and binary parsing script to ingest the raw IDX data files, circumventing common SSL certificate issues with high-level API wrappers.
2. **`src/features.py`**: I scale the raw pixel values from 0-255 down to 0.0-1.0 to ensure optimal convergence for the distance-based t-SNE algorithm.
3. **`src/model.py`**: I apply the t-SNE algorithm to reduce the 784 dimensions down to 2, saving the computed coordinates for downstream usage.
4. **`src/dashboard.py`**: I build an interactive Streamlit application via Plotly Express, allowing users to dynamically filter and explore the image clusters.

## Key Findings
Through this pipeline, I validated that t-SNE  outperforms linear techniques like PCA when handling complex, non-linear image data. While the initial 784-dimensional space is overwhelmingly sparse, t-SNE successfully captures local visual neighborhoods, naturally grouping footwear away from upper-body clothing without any prior knowledge of what these items look like.

## How to Run Locally

**1. Clone the repository and install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run the Pipeline:**
I execute the pipeline sequentially to fetch, process, and reduce the data:

```Bash
python -m src.data
python -m src.features
python -m src.model
```
**3. Launch the Dashboard**
```bash
streamlit run src/dashboard.py
```