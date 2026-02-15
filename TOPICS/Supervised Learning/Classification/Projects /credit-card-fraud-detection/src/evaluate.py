
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

def evaluate_model(model, X_test, y_test):
    """
    Generates a full evaluation suite for the model.
    """
    # 1. Generate Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] # Probability of being Class 1

    # 2. The Text Report
    # This gives us Precision, Recall, and F1-score
    print("\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred))

    # 3. Plotting the Results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Plot A: Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    # Note: Label 0=Normal, 1=Fraud

    # --- Plot B: Precision-Recall Curve ---
    # This is the best for imbalanced data(better than ROC)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    avg_prec = average_precision_score(y_test, y_probs)

    ax2.step(recall, precision, color='b', alpha=0.2, where='post')
    ax2.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    ax2.set_xlabel('Recall (Catch Rate)')
    ax2.set_ylabel('Precision (Accuracy of Flagging)')
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlim([0.0, 1.0])
    ax2.set_title(f'Precision-Recall Curve (AP={avg_prec:.2f})')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # This part allows us to test the script by pulling from our training script
    from src.train import train_and_evaluate
    model, X_test, y_test = train_and_evaluate()
    evaluate_model(model, X_test, y_test)