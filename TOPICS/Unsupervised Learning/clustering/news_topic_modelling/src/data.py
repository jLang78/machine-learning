import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path


# --- CONFIGURATION ---
# 1. find the .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2/top-headlines"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_and_save(category="technology"):
    # --- DEBUG CHECK ---
    if not API_KEY:
        print("ERROR: Could not find API Key.")
        print(f"  looked for the .env file here: {env_path}")
        print("   Ensure the file exists and contains NEWS_API_KEY=...")
        return

    print(f"Key found... (Starts with: {API_KEY[:4]}...)")
    print(f"Fetching {category} news...")

    # 2. Fetch
    params = {
        "country": "us",
        "category": category,
        "apiKey": API_KEY,
        "pageSize": 100
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        print(f" API Request Failed: {response.status_code}")
        print(response.json())
        return

    # 3. Save
    data = response.json()
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{timestamp}_{category}_news.json"
    filepath = RAW_DIR / filename

    # Ensure folder exists
    os.makedirs(RAW_DIR, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Success...Saved {len(data['articles'])} articles to:")
    print(f"   {filepath}")


if __name__ == "__main__":
    fetch_and_save()
