# src/features.py
import os
import sys
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def process_economic_features(input_filename="world_bank_live_raw.csv", output_dir="data/processed"):
    """
    I process the raw API data by coercing all values to numeric first,
    handling missing values, and engineering the polynomial features.
    """
    raw_path = os.path.join(project_root, "data", "raw", input_filename)
    processed_dir = os.path.join(project_root, output_dir)
    os.makedirs(processed_dir, exist_ok=True)

    print("I am loading raw API data for processing...")
    df = pd.read_csv(raw_path)

    print("I am coercing all API data to strict numeric formats...")
    # I isolate the economic feature columns (excluding country identifiers)
    feature_cols = [col for col in df.columns if col not in ['country_code', 'country_name']]

    # I force all economic columns to numeric, turning text strings into NaN
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("I am dropping regions with missing GDP data...")
    # Now that the data is numeric, I can safely drop rows where the target is NaN
    df_clean = df.dropna(subset=['gdp_per_capita']).copy()

    print("I am engineering polynomial and logarithmic features...")
    # I engineer the squared feature to capture the non-linear relationship
    df_clean['life_expectancy_squared'] = df_clean['life_expectancy'] ** 2

    # I apply the natural logarithm to the target variable
    df_clean['log_gdp_per_capita'] = np.log(df_clean['gdp_per_capita'])

    processed_path = os.path.join(processed_dir, "world_bank_processed.csv")
    df_clean.to_csv(processed_path, index=False)

    print(f"Feature engineering complete. Processed matrix saved with {len(df_clean)} rows.")


if __name__ == "__main__":
    process_economic_features()