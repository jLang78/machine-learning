# src/model.py
import os
import sys
import pandas as pd
import statsmodels.api as sm
import pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def train_and_save_model(input_filename="world_bank_processed.csv"):
    """
    I load the processed economic data, apply strict numeric coercion,
    and implement a two-stage imputation failsafe to guarantee the matrix
    never collapses to zero rows.
    """
    processed_path = os.path.join(project_root, "data", "processed", input_filename)
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)

    print("I am loading the processed data...")
    df = pd.read_csv(processed_path)

    features = [
        'trade_openness',
        'fdi_inflows',
        'capital_formation',
        'life_expectancy',
        'labor_force_part',
        'population_growth',
        'inflation_rate',
        'life_expectancy_squared'
    ]

    X = df[features].copy()
    y = df['log_gdp_per_capita'].copy()

    print("I am coercing all selected data strictly to numeric types...")
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    print("I am applying two-stage imputation to patch live API anomalies...")
    # Stage 1: I attempt to fill missing gaps with the column's mathematical median
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    # Stage 2: Failsafe. If the API returned a 100% missing column, the median is NaN.
    # I fill these catastrophic gaps with 0 so the matrix math can proceed.
    X = X.fillna(0)
    y = y.fillna(0)

    # I combine the features and target to ensure alignment
    model_data = pd.concat([y, X], axis=1)

    # I extract the finalized arrays
    y_clean = model_data['log_gdp_per_capita'].astype(float)
    X_clean = model_data.drop(columns=['log_gdp_per_capita']).astype(float)

    print(f"Verified matrix shape: {X_clean.shape[0]} rows and {X_clean.shape[1]} columns.")

    # I add a constant to the features for the statsmodels OLS intercept
    X_with_constant = sm.add_constant(X_clean)

    print("I am fitting the Multiple Linear Regression model...")
    model = sm.OLS(y_clean, X_with_constant).fit()

    print("Model training complete. I am saving the model artifact...")
    model_path = os.path.join(model_dir, "gdp_regression_model.pkl")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"I saved the model successfully to {model_path}")


if __name__ == "__main__":
    train_and_save_model()