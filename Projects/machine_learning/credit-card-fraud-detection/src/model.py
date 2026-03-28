
# This file defines what my model is. I keep it separate from training in order to
# easily swap in a different model (such as Random Forest) later without breaking the whole pipeline.

from src import config
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def get_baseline_model():
    """
    Returns a 'Dummy' model that always predicts the majority class (Normal).
    This serves as the floor for performance.
    """
    # strategy="most_frequent" meaning "Always predict 0"
    return DummyClassifier(strategy="most_frequent", random_state=config.RANDOM_STATE)


def get_logreg_model(balanced=True):
    """
    Returns a Logistic Regression model.

    Args:
        balanced (bool): If True, weighs the minority class (Fraud) higher
                         to handle the massive imbalance.
    """
    # If balanced is True, I use "balanced". If False, I use None.
    weight = "balanced" if balanced else None

    model = LogisticRegression(
        class_weight=weight,
        random_state=config.RANDOM_STATE,
        solver='lbfgs',  # Standard solver
        max_iter=1000  # Give it enough time to converge
    )
    return model



# ... now trying random forest clustering ...

def get_random_forest_model(balanced=True):
    """
    Returns a Random Forest model.
    This is generally more powerful than Logistic Regression for complex data.
    """
    weight = "balanced" if balanced else None

    model = RandomForestClassifier(
        n_estimators=100,  # Use 100 different 'trees'
        max_depth=10,  # Don't let trees grow too deep (prevents overfitting)
        class_weight=weight,  # Still handle the imbalance!
        random_state=config.RANDOM_STATE,
        n_jobs=-1  # Use all CPU cores to train faster
    )
    return model

# now trying gradient boosting
def get_boosting_model():
    """
    Returns a Histogram-based Gradient Boosting model (similar to LightGBM).
    Often faster and more accurate than Random Forest for big datasets.
    """
    # Note: HistGradientBoostingClassifier handles class weights differently.
    # It doesn't have a simple 'class_weight="balanced"' parameter in older versions,
    # but uses 'class_weight' in newer sklearn (v1.6+).
    # If using older sklearn, I rely on the inherent strength of boosting or manual weights.
    # However, standard GradientBoosting handles imbalanced data well naturally by focusing on hard cases.

    return HistGradientBoostingClassifier(
        learning_rate=0.1,  # How fast the model learns
        max_iter=100,  # The number of trees
        max_depth=10,  # The depth of trees
        random_state=config.RANDOM_STATE,
        scoring='f1',  # Optimising for F1 score (balance of precision/recall)
        class_weight='balanced'
    )

