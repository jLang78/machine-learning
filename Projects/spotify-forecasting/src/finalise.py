"""
The model has been trained on the past (pre-2021) and I've proved it works by testing it on 2021 data.

But this will not suffice. To predict streams for tomorrow (ie. in 2026), the model must be exposed to data from
2022, 2023, 2024, or 2025.

This script trains the model on 100% of the available data.

So I don't use Train/Test split, instead I use every single row available.
Because in production, I don't need to keep any data for testing the model, the efficacy was already proved during
the previous phase.
"""

import pandas as pd
import sys
import os

# Link project root
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from src import config
from src import features
from src import models

def finalise_model():
    """
    Trains the final production model on all available data.
    """
    print("\n" + "=" * 40)
    print("   PHASE: FINAL PRODUCTION TRAINING")
    print("=" * 40)

    # --- 1. Load ALL Data ---
    print("Loading 100% of available data...")
    df_raw = features.load_raw_data("ed_sheeran_charts.csv")
    df_processed = features.preprocess_data(df_raw)

    # I do not split by date. I use everything.
    feature_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'day_of_week', 'is_weekend', 'month']
    target_col = 'target'

    X = df_processed[feature_cols]
    y = df_processed[target_col]

    print(f"Training on full history: {X.shape[0]} days of data.")

    # --- 2. Train the 'production-ready' model ---
    print("Training Production Model...")
    model = models.get_model()

    # There is no 'eval_set' here because I'm using all data for training
    # I disable early stopping because we have no validation set
    # (i.e purposefully using 100% of data for the final build)
    model.set_params(early_stopping_rounds=None)
    # I rely on the hyperparameters validated during the previous phase.
    model.fit(X, y, verbose=False)

    print("Training Complete.")

    # --- 3. Saving the artifact ---
    model_dir = config.PROJ_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Giving it a name to indicate it's the live version
    save_path = model_dir / "xgboost_shape_of_you_production.json"

    model.save_model(save_path)

    print(f"Production Model saved to: {save_path}")
    print("Ready for inference!")


if __name__ == "__main__":
    finalise_model()


