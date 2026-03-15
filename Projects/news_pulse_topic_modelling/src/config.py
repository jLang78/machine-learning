import os
from pathlib import Path
from dotenv import load_dotenv

# ----- PATHS --
# This robustly finds the root of the project (two folders up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- ENVIRONMENT VARIABLES -
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# -- API SETTINGS ---
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
DEFAULT_CATEGORY = "technology"

# --- MODEL SETTINGS -----
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"