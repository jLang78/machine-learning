"""
While train.py builds the model, evaluate.py audits. It tests the saved model to ensure the file on
the hard drive actually works.

It loads the saved model (validating the JSON file).
It predicts on the test set (2021 data).
It visualises the results (actual vs. forecast).
It then saves the graph as an image file
"""
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # New import for formatting numbers
from sklearn.metrics import mean_absolute_error
import sys
import os

# Link project root
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from src import config
from src import features


def evaluate_model():
    """
    Generates a professional forecast report with residuals.
    """
    print("\n" + "=" * 40)
    print("   PHASE: MODEL EVALUATION & REPORTING")
    print("=" * 40)

    # --- 1. Load Test Data ---
    df_raw = features.load_raw_data("ed_sheeran_charts.csv")
    df_processed = features.preprocess_data(df_raw)

    df_processed = df_processed.set_index('date')
    cutoff_date = "2021-01-01"
    test = df_processed.loc[df_processed.index >= cutoff_date].copy()

    feature_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'day_of_week', 'is_weekend', 'month']
    target_col = 'target'

    X_test = test[feature_cols]
    y_test = test[target_col]

    # --- 2. Load Model ---
    model_path = config.PROJ_ROOT / "models" / "xgboost_shape_of_you.json"
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # --- 3. Forecast ---
    test['prediction'] = model.predict(X_test)
    test['error'] = test[target_col] - test['prediction']  # Calculate the difference

    mae = mean_absolute_error(test[target_col], test['prediction'])
    print(f"   Model MAE: {mae:,.0f} streams")

    # --- 4. Professional Plotting ---
    print("Generating Report...")

    # Create a figure with TWO plots (Main Forecast + Residuals)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

    # --- TOP PLOT: The Forecast ---
    ax1.plot(test.index, test[target_col], label='Actual Streams', color='#1f77b4', alpha=0.7)
    ax1.plot(test.index, test['prediction'], label='AI Forecast', color='#d62728', linewidth=2)

    ax1.set_title(f"2021 Forecast: Shape of You (Global)\nAverage Error: +/- {mae:,.0f} streams", fontsize=16)
    ax1.set_ylabel("Daily Streams", fontsize=12)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Format Y-axis with commas (e.g. 1,000,000)
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # --- Plot the residuals ---
    # This shows me if the model is consistently guessing too high or too low
    ax2.fill_between(test.index, test['error'], color='gray', alpha=0.3, label='Error Gap')
    ax2.plot(test.index, test['error'], color='gray', alpha=0.8)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)  # The "Perfect" line

    ax2.set_ylabel("Prediction Error", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # Save
    report_dir = config.PROJ_ROOT / "reports" / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    save_file = report_dir / "forecast_performance_pro.png"

    plt.tight_layout()
    plt.savefig(save_file)
    print(f"✅ Professional Report saved to: {save_file}")


if __name__ == "__main__":
    evaluate_model()
