# Unsupervised Market Discovery: News Topic Modeling

I developed this system to automatically categorise raw news streams into distinct topics using Natural Language Processing. This project serves to prove that we can extract structured insights from thousands of unstructured articles without human intervention.

## Value
In a high-velocity information environment, manually tagging articles is too slow. I built this pipeline to help identify "Topic Drift" and emerging trends in real-time, providing a competitive edge in market sentiment analysis.

## Core Technology
I utilised the NewsData.io API to fetch live articles and implemented the following:
* **Preprocessing:** I used NLTK for tokenisation, lemmatization, and stop-word removal.
* **Algorithm:** I chose Latent Dirichlet Allocation (LDA) to discover hidden thematic structures.
* **Dimensionality Reduction:** I used t-SNE to visualise the "distance" between different news topics in a 2D space.

## Project Structure
* **src/ingestion.py**: Handles API authentication and fetches the latest 5,000 headlines.
* **src/processing.py**: Converts raw text into a Bag-of-Words (BoW) and TF-IDF matrix.
* **src/model.py**: Trains the LDA model and evaluates topic coherence scores.

## How to Run
1. Obtain an API key from [NewsData.io].
2. Create a `.env` file in the root directory and add `API_KEY=your_key_here`.
3. Run the pipeline:
   ```bash
   python -m src.ingestion
   python -m src.model





