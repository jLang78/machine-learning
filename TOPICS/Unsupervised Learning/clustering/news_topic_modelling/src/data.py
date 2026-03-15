import sys
from pathlib import Path
import json
import requests
from datetime import datetime

# I explicitly add the project root to the system path so Python can find the 'src' module.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# I import all my settings directly from the new config file.
from src.config import NEWS_API_KEY, NEWS_API_URL, RAW_DATA_DIR, DEFAULT_CATEGORY


def fetch_and_save(category=DEFAULT_CATEGORY):
    # I check if the API key is loaded properly before proceeding.
    if not NEWS_API_KEY:
        print("ERROR: API Key missing in .env file.")
        return

    print(f"Fetching {category} news...")

    # I set up the parameters for the API request.
    params = {
        "country": "us",
        "category": category,
        "apiKey": NEWS_API_KEY,
        "pageSize": 100
    }

    # I make the GET request to the News API.
    response = requests.get(NEWS_API_URL, params=params)

    # I handle any errors returned by the API.
    if response.status_code != 200:
        print(f"API Request Failed: {response.status_code}")
        return

    # I parse the JSON response into a Python dictionary.
    data = response.json()

    # I generate a timestamped filename so I do not overwrite older data.
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{timestamp}_{category}_news.json"
    filepath = RAW_DATA_DIR / filename

    # I ensure the raw directory exists before attempting to save the file.
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # I write the JSON data to my local storage.
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Success! Saved {len(data.get('articles', []))} articles to: {filepath}")


if __name__ == "__main__":
    fetch_and_save()