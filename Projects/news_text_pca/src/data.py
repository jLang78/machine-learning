# src/data.py
import requests
import json
import os
import sys
from datetime import datetime


# Get the absolute path of the directory one level up from this file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to the system path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the key from the config file
from src.config import NEWS_API_KEY


def fetch_news_data(queries, output_dir="data/raw"):
    """
    Fetches news articles for a list of query terms and saves them to a JSON file.
    """
    # Ensure the target directory exists
    os.makedirs(output_dir, exist_ok=True)

    all_articles = []

    for query in queries:
        print(f"Fetching news for category: '{query}'...")

        # NewsAPI 'Everything' endpoint
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={query}&"
            f"language=en&"
            f"sortBy=relevancy&"
            f"pageSize=100&"  # Pull 100 articles per category
            f"apiKey={NEWS_API_KEY}"
        )

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            # Inject our query term as a "category" label so we can color-code our PCA plot later!
            for article in data.get("articles", []):
                article["category"] = query
                all_articles.append(article)
        else:
            print(f"Failed to fetch '{query}'. Status code: {response.status_code}")
            print(response.text)

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"news_raw_{timestamp}.json")

    # Dump the combined articles into a JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=4)

    print(f"\nSuccess! Saved {len(all_articles)} total articles to {filename}")
    return filename


if __name__ == "__main__":

    topics = ["technology", "sports", "finance", "cooking"]

    # Run the fetcher
    fetch_news_data(topics)