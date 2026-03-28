
"""This script defines the model. Instead of typing xgb.XGBRegressor(...) every time I want to train,
I save the configuration here.

If later I switch from XGBoost to a different model (like LightGBM or Random Forest),
I need only change it in this one file, and the whole project updates automatically."""

import xgboost as xgb

def get_model():
    """
    Returns the configured XGBoost Regressor.

    Configuration:
    - n_estimators=1000: High number of trees (will be stopped early if needed)
    - learning_rate=0.01: Slow learning for higher accuracy
    - max_depth=5: Medium complexity to prevent overfitting
    """
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        early_stopping_rounds=50,
        objective='reg:squarederror',
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )
    return model
