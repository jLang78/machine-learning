"""
This script  answers the question of "How many streams will Ed Sheeran get tomorrow?"

To predict tomorrow, the model needs inputs:
E.g
"What was the streaming count yesterday?"
"What is the 7-day trend?"
"Is tomorrow a Friday?"

It needs more than a date. I have to construct these features based on the very last day of data we have.

1) The model loads the full dataset (the 'history')

2) It finds the last date in the file and creates a new row for the "Next Day."

3)  It runs the preprocess_data function on this new expanded dataset. This automatically calculates
the correct lags (e.g., "Yesterday's Streams") for the new day.

4) It loads the Production Model and predicts the streams for that single new day.
"""

import pandas as pd
import xgboost as xgb
import sys
import os

# Link project root
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from src import config
from src import features


def predict_next_day():
    """
    Generates a forecast for the 'Next Day' after the dataset ends.
    """
    print("\n" + "=" * 40)
    print("LIVE INFERENCE SYSTEM")
    print("=" * 40)

    # --- 1. Load Recent History ---
    print("Loading history...")
    df_raw = features.load_raw_data("ed_sheeran_charts.csv")

    # Sort and clean to find the absolute last day
    df_clean = df_raw[
        (df_raw['title'] == "Shape of You") &
        (df_raw['region'] == "Global")
        ].sort_values(['date', 'streams'], ascending=[True, False])

    df_clean = df_clean.drop_duplicates(subset=['date'], keep='first')

    last_date = df_clean['date'].max()
    next_date = last_date + pd.Timedelta(days=1)

    print(f"Last Data Point: {last_date.date()}")
    print(f"Target Forecast Date: {next_date.date()}")

    # --- 2. Create row with empty values ---
    # I add a row for tomorrow with 'streams = NaN'
    # This allows the feature engineering model to calculate lags for it naturally
    future_row = pd.DataFrame([{
        'date': next_date,
        'title': "Shape of You",
        'region': "Global",
        'streams': 0,  # Placeholder (won't be used for lags)
        'artist': "Ed Sheeran"
    }])

    # Attach future row to history
    df_augmented = pd.concat([df_clean, future_row], ignore_index=True)

    # --- 3. Feature Engineering ---
    # This fills in 'lag_1', 'lag_7', etc. for the new row
    print("️Constructing features...")
    df_processed = features.preprocess_data(df_augmented)

    # Grabbing only the last row (the target date)
    # I must ensure I have the exact same columns the model expects
    feature_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'day_of_week', 'is_weekend', 'month']

    # The last row of the processed data is the 'Next Day'
    input_row = df_processed.iloc[[-1]][feature_cols]

    print("\n   --- Input Features for Model ---")
    print(input_row.to_string(index=False))

    # --- 4. Load Production Model ---
    model_path = config.PROJ_ROOT / "models" / "xgboost_shape_of_you_production.json"
    print(f"\nLoading Production Brain: {model_path.name}")

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # --- 5. Predict ---
    prediction = model.predict(input_row)[0]

    print("\n" + "-" * 30)
    print(f"FORECAST for {next_date.date()}")
    print(f"   {prediction:,.0f} Streams")
    print("-" * 30)


if __name__ == "__main__":
    predict_next_day()