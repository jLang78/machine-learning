# src/features.py
import os
import sys
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# --- get path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ----------------

def clean_text(text):
    """
    Lowercases text and removes punctuation/special characters.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_and_vectorize(json_filename, output_dir="data/processed", model_dir="models"):
    """
    Loads raw JSON, cleans text, applies TF-IDF, and saves the outputs.
    """
    # 1. Load the raw JSON data
    raw_path = os.path.join(project_root, "data", "raw", json_filename)
    print(f"Loading data from {raw_path}...")

    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Extract relevant fields into a Pandas DataFrame
    records = []
    for article in data:
        # Combine title and description for richer text context
        raw_text = str(article.get("title", "")) + " " + str(article.get("description", ""))
        records.append({
            "category": article.get("category", "unknown"),
            "raw_text": raw_text
        })

    df = pd.DataFrame(records)

    # 3. Clean the text
    print("Cleaning text...")
    df["cleaned_text"] = df["raw_text"].apply(clean_text)

    # Drop empty rows (sometimes articles have no title/description)
    df = df[df["cleaned_text"].str.strip() != ""]

    # 4. Initialise and fit the TF-IDF Vectorizer
    print("Vectorizing text with TF-IDF...")
    # max_features limits the dimensions to the top 2000 most important words
    # stop_words='english' automatically removes common words like 'the', 'and', 'is'
    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")

    tfidf_matrix = vectorizer.fit_transform(df["cleaned_text"])

    print(f"Created a TF-IDF matrix with shape: {tfidf_matrix.shape}")
    # Shape will be (Number of Articles, 2000) -> These 2000 dimensions are what PCA will shrink!

    # 5. Save everything for the next steps
    os.makedirs(os.path.join(project_root, output_dir), exist_ok=True)
    os.makedirs(os.path.join(project_root, model_dir), exist_ok=True)

    # Save the DataFrame (labels and text)
    df_path = os.path.join(project_root, output_dir, "processed_news.csv")
    df.to_csv(df_path, index=False)

    # Save the TF-IDF matrix
    matrix_path = os.path.join(project_root, output_dir, "tfidf_matrix.pkl")
    joblib.dump(tfidf_matrix, matrix_path)

    # Save the Vectorizer model
    vectorizer_path = os.path.join(project_root, model_dir, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Saved processed data to {df_path}")
    print(f"Saved TF-IDF matrix to {matrix_path}")
    print(f"Saved TF-IDF vectorizer to {vectorizer_path}")


if __name__ == "__main__":

    recent_json_file = "news_raw_20260216_140910.json"

    try:
        process_and_vectorize(recent_json_file)
    except FileNotFoundError:
        print(
            "Error: Please update 'recent_json_file' at the bottom of the script with your actual filename from data/raw/")

