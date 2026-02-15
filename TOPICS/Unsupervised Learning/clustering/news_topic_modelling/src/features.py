import re
import pandas as pd


def clean_news_text(text):
    """
    Cleans headlines and descriptions for topic modeling.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Removeing URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 2. Removing 'Read more' / '... [+ chars]' patterns (common in NewsAPI)
    text = re.sub(r'\[\+\d+\s+chars\]', '', text)

    # 3. Removing the source attribution (e.g., " - CNN")
    text = re.sub(r'\s-\s.*$', '', text)

    # 4. Removing special characters and numbers (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 5. Converting to lowercase and stripping whitespace
    return text.lower().strip()


def process_raw_to_clean(df):
    """
    Takes a dataframe of articles and returns a cleaned version.
    """
    # Create a 'full_content' column combining title and description
    # This gives the topic model more context to work with.
    df['text_combined'] = df['title'].fillna('') + " " + df['description'].fillna('')

    # Apply cleaning
    df['text_cleaned'] = df['text_combined'].apply(clean_news_text)

    # Remove rows where cleaning resulted in nearly empty strings
    df = df[df['text_cleaned'].str.len() > 20]

    return df


if __name__ == "__main__":
    # Quick test to verify cleaning logic
    test_text = "AI is taking over! (Read more at http://tech.com) - The Verge [+400 chars]"
    print(f"Original: {test_text}")
    print(f"Cleaned:  {clean_news_text(test_text)}")