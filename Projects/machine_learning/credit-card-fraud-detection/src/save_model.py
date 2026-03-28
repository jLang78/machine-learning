import joblib
from src.train import train_and_evaluate
from src import config


def save_final_model():
    print("Training Final Random Forest Model...")
    model, _, _ = train_and_evaluate()

    # Define the save path
    save_path = config.PROJ_ROOT / "models" / "model.joblib"

    # Ensure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the file
    print(f"Saving model to {save_path}...")
    joblib.dump(model, save_path)
    print("✅ Model saved successfully!")


if __name__ == "__main__":
    save_final_model()