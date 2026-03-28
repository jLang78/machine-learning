# News Pulse: Unsupervised Topic Modeling Pipeline

### What This Project Is
I built this end-to-end Machine Learning pipeline to automatically ingest unstructured technology news, process the raw text, and mathematically discover hidden themes without any human labeling or manual categorization. 

### What It's For
Organisations have to deal with unstructured text—customer reviews, support tickets, and social media mentions. No human has the time to read thousands of documents to figure out what people are talking about. I designed this project to prove that an Unsupervised Machine Learning system can automatically read, group, and track evolving narratives (Topic Drift) at scale.

### How It Does It
This pipeline utilises **BERTopic**, which leverages Deep Learning and state-of-the-art Natural Language Processing (NLP). 
1. **Embeddings:** I use a lightweight Hugging Face transformer (`all-MiniLM-L6-v2`) to convert English sentences into numerical vectors, capturing the semantic meaning of the text.
2. **Clustering:** I use HDBSCAN to group these vectors together in multidimensional space.
3. **Extraction:** I use c-TF-IDF to extract the most important keywords to "name" the clusters.

### Why This Works (The Output)
Because the embedding model understands semantic relationships (e.g., it knows "Apple" and "iPhone" are related), it successfully segments the news without me ever defining the categories. The output is an interactive Streamlit dashboard that provides:
* **Topic Drift (Time Series):** A line chart tracking the volume of specific topics over time.
* **Top Topics (Bar Chart):** The specific c-TF-IDF keyword DNA that makes up each cluster.
* **Topic Clusters (Map):** A 2D representation of the semantic distance between different news themes.
---

### Project Structure

Prototyping (Notebooks)

I used Jupyter Notebooks as my "rough draft" environment to experiment with the data and prove the machine learning logic before building the final application.

* `notebooks/01_data_collection.ipynb`: I prototyped the API connection and raw data parsing here.
* `notebooks/02_modeling.ipynb`: I experimented with text cleaning and the core BERTopic clustering algorithm.
* `notebooks/03_drift_analysis.ipynb`: I developed the time-series logic to calculate how topics evolve across discrete days.

* I designed this codebase using modular, production-ready software engineering principles.

* `src/config.py`: I centralised all file paths, API endpoints, and environment variables here so the codebase is easy to maintain.
* `src/data.py`: I wrote this script to connect to the NewsAPI, pull the latest technology headlines, and save them as timestamped JSON files in a data lake architecture.
* `src/features.py`: I built this cleaning pipeline to strip out URLs, special characters, and journalistic boilerplate using regular expressions.
* `src/model.py`: I encapsulated the BERTopic training logic and the time-series drift calculations into reusable functions here.
* `src/visualise.py`: I isolated the Plotly graphing logic into this file so the frontend application can call the charts on demand.
* `src/dashboard.py`: I built the frontend user interface using Streamlit. It glues the pipeline together, allowing users to train the model and filter the visual output interactively.

---

### How to Run This Project Locally

**Note:** To keep the repository lightweight and respect data privacy, the `/data/` folder is ignored by Git. You must fetch your own data to run the dashboard.

**1. Clone the repository**
```bash
git clone <your-repository-url>
cd news_topic_modelling
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**
Create a `.env` file in the root directory and add your API keys:
```text
NEWS_API_KEY=your_newsapi_key_here
HF_TOKEN=your_huggingface_token_here
```

**4. Fetch the data**
Because the repository does not include my historical JSON files, you must run the data ingestion script to pull today's news. (For time-series drift, run this once a day for a few days).
```bash
python src/data.py
```

**5. Launch the Dashboard**
```bash
streamlit run src/dashboard.py
```

---

### Future Model Improvements

While this prototype successfully identifies distinct themes like gaming console rivalries and smartphone ecosystems, I would implement the following upgrades to scale this into an enterprise-grade application:

1. **Increase Data Volume:** The model currently runs on a few hundred articles. Unsupervised Deep Learning requires density. I would connect this to a paid API tier or build custom RSS scrapers to ingest tens of thousands of documents, which would solidify the clusters and reduce noise.
2. **Upgrade the Embedding Model:** I am currently using `all-MiniLM-L6-v2` for fast local prototyping. For production, I would swap this for a heavier, state-of-the-art open-source model (like `bge-large-en-v1.5`) or a commercial API (like OpenAI) for superior semantic understanding of modern tech jargon.
3. **Hyperparameter Tuning (HDBSCAN):** I currently have the minimum cluster size set very low (`min_topic_size=5`) to accommodate the small dataset. In a larger dataset, I would increase this to 50 or 100 to force the algorithm to only report on massive, undeniable macro-trends.
4. **Custom Stop Words:** I am currently using standard English stop words. I would build a custom journalistic dictionary to force the `CountVectorizer` to ignore filler words like "breaking", "report", and "update", focusing the AI entirely on core entities.




