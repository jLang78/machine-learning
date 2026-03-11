
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src import config


def load_and_split_data(test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """
    Loads raw data, splits it into Train/Test, and scales numerical features.

    Returns:
        X_train, X_test, y_train, y_test (tuples of DataFrames/Series)
    """
    # 1. Load Data using our existing tool
    # Local import prevents circular dependency errors
    from src.data import load_raw_data
    df = load_raw_data()

    # 2. Separate Features (X) and Target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 3. Split into Train and Test
    # 'stratify=y' is critical as it ensures the fraud ratio is preserved in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4. Drop 'Time' (Noise for a baseline model)
    if 'Time' in X_train.columns:
        X_train = X_train.drop('Time', axis=1)
        X_test = X_test.drop('Time', axis=1)

    # 5. Scaling
    # We initialize the scaler
    scaler = StandardScaler()

    # We fit the scaler ONLY on X_train.
    # Then we use those learned stats (mean/std) to transform X_test.
    # This prevents "Data Leakage" (cheating by seeing the test answers).

    # Note: V-features are already PCA-scaled, but 'Amount' is not.
    # Scaling everything is the safest bet for Logistic Regression convergence.
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for readability (so we keep column names)
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # This block runs if you execute the script directly
    X_tr, X_te, y_tr, y_te = load_and_split_data()

    print("✅ Data Processed Successfully")
    print("-" * 30)
    print(f"Train Matrix: {X_tr.shape}")
    print(f"Test Matrix:  {X_te.shape}")
    print("-" * 30)
    print(f"Train Fraud Rate: {y_tr.mean():.4%}")
    print(f"Test Fraud Rate:  {y_te.mean():.4%}")
    print("-" * 30)
    print("Sample Scaled Data (First 3 rows):")
    print(X_tr.iloc[:3, :3])  # Show first 3 rows and 3 columns