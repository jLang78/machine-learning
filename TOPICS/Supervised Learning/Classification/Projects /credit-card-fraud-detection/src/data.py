import pandas as pd
from pathlib import Path
from src import config


def load_raw_data():
    """
    Loads the creditcard.csv from the raw data directory.

    Returns:
        pd.DataFrame: The raw credit card data.
    """
    file_path = config.DATA_RAW / "creditcard.csv"

    # Safety check
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at {file_path}. "
                                f"Please download it from Kaggle and put it in data/raw/")

    # Load data
    df = pd.read_csv(file_path)
    return df


if __name__ == "__main__":
    # This block runs only when I execute this script directly
    # It allows me to test if the data loads correctly without opening a notebook
    try:
        df = load_raw_data()
        print(f"✅ Data loaded successfully!")
        print(f"Shape: {df.shape} (Rows, Columns)")

        # Quick check for the target column
        if 'Class' in df.columns:
            fraud_count = df['Class'].sum()
            total_count = len(df)
            print(f"Fraud cases: {fraud_count}")
            print(f"Fraud rate: {fraud_count / total_count:.4%}")
        else:
            print("⚠️ Warning: 'Class' column not found.")

    except Exception as e:
        print(f"❌ Error: {e}")

"""I can see that the fraud rate is roughly 0.17%, therefore 99.83% of this data is non-fraudulent
The goal should be to beat this baseline and find the 0.17% of frauds. The questions should be:
1) Of the total number of fraudulent transactions, how many did the model catch?
2) When a transaction is flagged as fraud, how often is this corect?"""
