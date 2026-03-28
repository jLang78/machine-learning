

"""
This script will contain the logic to turn raw data into production ready features.
In essence, this means taking the code from the 03_feature_eng.ipynb notebook and wrapping it
in a function called preprocess_data.

In a notebook, changing the lag from 7 days to 14 days, requires scrolling
and re-running cells. Here, I need only change one argument.
"""

import pandas as pd
import numpy as np
from src import config


def load_raw_data(file_name: str = "ed_sheeran_charts.csv"):
    """
    Loads the raw CSV from the data/raw directory.
    """
    path = config.PROJ_ROOT / "data" / "raw" / file_name
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def preprocess_data(df: pd.DataFrame, target_song="Shape of You", target_region="Global"):
    """
    The feature engineering pipeline:
    1. Filter for specific song/region
    2. Remove duplicates
    3. Create Lags and Temporal Features
    4. Handle NaN values
    """
    print(f" Processing data for: {target_song} ({target_region})")

    # 1. Filter
    df_model = df[
        (df['title'] == target_song) &
        (df['region'] == target_region)
        ].copy()

    # 2. De-Duplicate (Keep highest streams per day)
    df_model = df_model.sort_values(['date', 'streams'], ascending=[True, False])
    df_model = df_model.drop_duplicates(subset=['date'], keep='first')
    df_model = df_model.sort_values('date').reset_index(drop=True)

    # 3. Create Lag Features (The Signal)
    # Target: Tomorrow's streams
    df_model['target'] = df_model['streams'].shift(-1)

    # Lags: Past streams
    df_model['lag_1'] = df_model['streams'].shift(1)
    df_model['lag_7'] = df_model['streams'].shift(7)
    df_model['rolling_mean_7'] = df_model['streams'].rolling(window=7).mean()

    # 4. Temporal Features (The Context)
    df_model['day_of_week'] = df_model['date'].dt.dayofweek
    df_model['month'] = df_model['date'].dt.month
    df_model['is_weekend'] = df_model['day_of_week'].isin([5, 6]).astype(int)

    # 5. Clean NaNs
    df_model = df_model.dropna()

    # Select only the columns we need for the model
    features = ['date', 'lag_1', 'lag_7', 'rolling_mean_7', 'day_of_week', 'is_weekend', 'month', 'target']

    print(f"Data Processed. Final Shape: {df_model.shape}")
    return df_model[features]