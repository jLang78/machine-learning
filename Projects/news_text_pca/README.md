# Dimensionality Reduction: News Topic Clustering

## Project Overview
This project demonstrates the use of Unsupervised Machine Learning - specifically **Principal Component Analysis (PCA)** and **TF-IDF** - to automatically process, vectorise, and cluster live news articles based on their textual content. 

The goal is to reduce high-dimensional text data into a 2D space to visualise semantic relationships and group similar topics without manual human tagging.

## Value
Unsupervised clustering of text data provides several high-impact business applications:
* **Automated Categorisation:** Automatically routing or tagging thousands of daily incoming documents, customer support tickets, or news feeds based on mathematical similarity.
* **Content Recommendation:** Enhancing user retention by recommending articles that are closest in proximity within the PCA vector space.
* **Anomaly Detection:** Identifying outliers in text data (e.g., detecting spam or off-topic submissions) by finding data points that fall far outside standard topical clusters.

## Exploration & Analysis (Jupyter Notebooks)
Before building the automated pipeline, the data and math were prototyped in Jupyter Notebooks to ensure statistical validity:
* **`01_api_exploration.ipynb`**: Demonstrates the initial secure connection to the NewsAPI, JSON parsing, and pandas DataFrame prototyping.
* **`02_tfidf_and_pca.ipynb`**: Contains mathematical evaluation of the PCA model. Includes a **Scree Plot** which visually proves the inherent sparsity of text data; demonstrating that while 2 components are suitable for visualization, hundreds of components are required to capture >80% of the dataset's cumulative variance.


## Project Architecture
This project is structured as a production-ready pipeline rather than a static notebook.

1. **`src/data.py`**: Calls the live NewsAPI to fetch up-to-date articles across distinct categories (Technology, Sports, Finance, etc.).
2. **`src/features.py`**: Cleans the raw JSON data and applies a `TfidfVectorizer` to convert English text into a 2,000-dimensional mathematical matrix.
3. **`src/model.py`**: Applies **PCA (Principal Component Analysis)** to crush the 2,000 sparse dimensions down to 2 dense principal components (X and Y coordinates).
4. **`src/dashboard.py`**: A local Streamlit web application utilising Plotly Express to render the clusters in an interactive, user-friendly UI.

## Key Findings & Analysis
The integration of our automated pipeline and deep-dive notebooks yielded several critical insights:
* **The Sparsity of Language:** The Scree Plot analysis revealed that even 50 principal components only capture ~30% of the total variance. This is a nice quantifier of the "curse of dimensionality" in NLP; because vocabulary is so diverse, information is spread thinly across thousands of features rather than being concentrated in a few.
* **Linear Clustering vs. Semantic Density:** The PCA visualisation successfully isolated "Outlier" topics....specifically Sports and Cooking, where the vocabulary is highly specialised (e.g., "recipe," "match," "league"). However, Technology and Finance exhibited significant overlap in the 2D plane. This suggests that while PCA is excellent for global variance and identifying distinct outliers, it struggles with the non-linear, nuanced semantic relationships found in standard prose.
* **Architectural Scalability:** By moving from a notebook-only workflow to a modular `src/` architecture, I separated the sandbox (exploration) from engineering (production). This ensures that the pipeline can be easily updated with more complex models (like t-SNE or UMAP) without refactoring the entire data ingestion engine.


## How to Run Locally

**1. Clone the repository and install dependencies:**
```bash
pip install -r requirements.txt


