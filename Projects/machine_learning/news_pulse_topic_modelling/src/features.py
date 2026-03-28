
import re
import pandas as pd


def clean_news_text(text):
    # I handle empty or invalid text inputs to prevent errors.
    if not text or not isinstance(text, str):
        return ""

    # removing URLs from the text.
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # i strip out the specific '[+ chars]' pattern common in the NewsAPI.
    text = re.sub(r'\[\+\d+\s+chars\]', '', text)

    # deleting the source attribution that often appears at the end of headlines.
    text = re.sub(r'\s-\s.*$', '', text)

    #removing all special characters and numbers, keeping only letters and spaces.
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # converting the string to lowercase and remove leading/trailing whitespace.
    return text.lower().strip()


def process_raw_to_clean(df):
    # I combine the title and description columns to provide more context to the model.
    df['text_combined'] = df['title'].fillna('') + " " + df['description'].fillna('')

    # I apply my cleaning function to the new combined column.
    df['text_cleaned'] = df['text_combined'].apply(clean_news_text)

    # I filter out any rows where the resulting cleaned text is too short to be useful.
    # create a distinct copy to avoid Pandas SettingWithCopy warnings later in the pipeline.
    df = df[df['text_cleaned'].str.len() > 20].copy()

    return df


if __name__ == "__main__":
    #  running a quick test to verify my logic if this script is executed directly.
    test_text = "BREAKING: AI is taking over! (Read more at http://tech.com) - The Verge [+400 chars]"
    print(f"Original: {test_text}")
    print(f"Cleaned:  {clean_news_text(test_text)}")