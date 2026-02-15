
# This file is the conductor. It imports both the data and the model, and consolidates them for training
import sys
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.features import load_and_split_data
from src.model import get_baseline_model, get_logreg_model
from src.model import get_random_forest_model
from src.model import get_boosting_model


def train_and_evaluate():
    # 1. Load Data
    print("Loading and Splitting Data...")
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"   Data Loaded. Train shape: {X_train.shape}")

    # ==========================================
    # The Baseline (Dummy)
    print("\n Training Baseline (Dummy) Model...")
    dummy = get_baseline_model()
    dummy.fit(X_train, y_train)

    # Quick check on Train set
    dummy_acc = dummy.score(X_train, y_train)
    print(f"   Baseline Accuracy (Train): {dummy_acc:.4%} (This is the 'Guess Normal' strategy)")

    # ==========================================
    # The Real Model (this can be changed depending on the ML algorithm - Random Forest was found to outperform others)
    print("\n Training Random Forest Model...this could take a few seconds")
    # We use balanced=True because our EDA showed huge imbalance
    model = get_random_forest_model(balanced=True)
    model.fit(X_train, y_train)

    print("   Training Complete.")

    # --- Initial Check (Not full evaluation yet) ---
    # I check the training accuracy just to make sure it learned something
    train_acc = model.score(X_train, y_train)
    print(f"   Model Accuracy (Train): {train_acc:.4%}")

    # Return the trained model and data for further steps if needed
    return model, X_test, y_test


if __name__ == "__main__":
    train_and_evaluate()