# src/config.py
import os
from dotenv import load_dotenv

# Find + load the .env file
load_dotenv()

# fetch API key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# quick safety check
if not NEWS_API_KEY:
    raise ValueError("NEWS_API_KEY is not set. Please check your .env file.")