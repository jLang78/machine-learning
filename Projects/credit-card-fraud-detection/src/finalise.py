import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import pandas as pd
import joblib
from src import config
from src.features import load_and_split_data


def finalize_project():
    print("  Starting Final Project Cleanup...")

    # 1. Load Data & Model
    print("   Loading data and trained model...")
    _, X_test, _, y_test = load_and_split_data()

    model_path = config.PROJ_ROOT / "models" / "model.joblib"
    if not model_path.exists():
        print("   ❌ Model not found! Please run 'python -m src.save_model' first.")
        return
    model = joblib.load(model_path)

    # 2. Populate 'data/processed'
    # Creating a test set so anyone can verify my results later without re-running the split
    print("  Saving processed test data...")
    save_dir_data = config.PROJ_ROOT / "data" / "processed"
    save_dir_data.mkdir(parents=True, exist_ok=True)

    X_test.to_csv(save_dir_data / "X_test.csv", index=False)
    y_test.to_csv(save_dir_data / "y_test.csv", index=False)
    print(f"      Saved to {save_dir_data}")

    # 3. Populate 'reports/figures'
    print("   Generating and saving final reports...")
    save_dir_figs = config.PROJ_ROOT / "reports" / "figures"
    save_dir_figs.mkdir(parents=True, exist_ok=True)

    # Generate Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # --- Plot 1: Confusion Matrix ---
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Final Confusion Matrix (Random Forest)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir_figs / "confusion_matrix.png")
    plt.close()  # Close to prevent popping up

    # --- Plot 2: Precision-Recall Curve ---
    plt.figure(figsize=(6, 5))
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    avg_prec = average_precision_score(y_test, y_probs)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP={avg_prec:.2f})')
    plt.tight_layout()
    plt.savefig(save_dir_figs / "precision_recall_curve.png")
    plt.close()

    print(f"      Saved charts to {save_dir_figs}")
    print("✅ Project Finalization Complete.")


if __name__ == "__main__":
    finalize_project()