"""
This script orchestrates the entire pipeline. When run, it will:

Ingest: Call features.py to get fresh, clean data.

Split: Separate the "past" (training) from the "future" (test).

Train: Call models.py to initialize the XGBoost brain and teach it.

Evaluate: Immediately calculate the error (MAE/RMSE) to see if the training worked.

Serialise: Save the trained model to the models/ folder so it can be used later without retraining.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path to ensure imports work when running from terminal
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from src import config
from src import features
from src import models


def train_pipeline():
    """
    Executes the complete training pipeline:
    Load - Process - Split - Train - Evaluate - Save
    """
    # --- 1. DATA INGESTION & PROCESSING ---
    print("\n" + "=" * 40)
    print("   PHASE 1: DATA INGESTION")
    print("=" * 40)

    # Load raw data
    df_raw = features.load_raw_data("ed_sheeran_charts.csv")

    # Processing into features
    df_processed = features.preprocess_data(
        df_raw,
        target_song="Shape of You",
        target_region="Global"
    )

    # --- 2. TEMPORAL SPLIT ---
    print("\n" + "=" * 40)
    print("   PHASE 2: TEMPORAL SPLIT")
    print("=" * 40)

    # Setting the index to date for slicing
    df_processed = df_processed.set_index('date')

    # Define the cutoff (what I'm calling the 'now')
    cutoff_date = "2021-01-01"

    # Split: train (Pre-2021) and test (2021)
    train = df_processed.loc[df_processed.index < cutoff_date].copy()
    test = df_processed.loc[df_processed.index >= cutoff_date].copy()

    # Defining feature matrix (X) and target vector (y)
    feature_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'day_of_week', 'is_weekend', 'month']
    target_col = 'target'

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    print(f"Training Data: {X_train.shape[0]} days (Ends {train.index.max().date()})")
    print(f"Test Data:     {X_test.shape[0]} days (Starts {test.index.min().date()})")

    # --- 3. MODEL TRAINING ---
    print("\n" + "=" * 40)
    print("   PHASE 3: TRAINING")
    print("=" * 40)

    # Initialise the model
    model = models.get_model()

    # Fit the model
    # I pass the test set as 'eval_set' to monitor performance live
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100  # Print update every 100 rounds
    )

    print("Model successfully trained.")

    # --- 4. EVALUATION ---
    print("\n" + "=" * 40)
    print("   PHASE 4: EVALUATION")
    print("=" * 40)

    # Predict
    predictions = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"Mean Absolute Error (MAE): {mae:,.0f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.0f}")

    # Logic check: Is this good enough?
    if mae < 25000:
        print("RESULT: EXCELLENT PERFORMANCE (Ready for Production)")
    else:
        print("RESULT: NEEDS IMPROVEMENT")

    # --- 5. SERIALISATION (SAVING) ---
    print("\n" + "=" * 40)
    print("   PHASE 5: SAVING ARTIFACTS")
    print("=" * 40)

    # Define save path
    model_dir = config.PROJ_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "xgboost_shape_of_you.json"

    # Saving using XGBoost's native format
    model.save_model(save_path)

    print(f"Model saved to: {save_path}")



if __name__ == "__main__":
    train_pipeline()
