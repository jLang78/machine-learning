"""ATP Tennis Linear Regression
This script follows a notebook-like workflow to keep things explainable."""

import pandas as pd  # for data handling
import matplotlib.pyplot as plt # for plotting
from itertools import combinations # provides utilities for trying feature combinations

# then import Scikit-learn tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# add path handling for file portability
from pathlib import Path
# -----------------------------
atp_stats = pd.read_csv('tennis_stats.csv')
print("\n=== DATA LOADED ===")
print("Shape (rows, columns):", atp_stats.shape)
print("\nFirst 5 rows:")
print(atp_stats.head(5))

# =================================
# 1) EDA (Exploratory Data Analysis)
print("\n EDA BASIC INFO")
print(atp_stats.info())

# selecting only numeric columns for data analysis
numeric: pd.DataFrame = atp_stats.select_dtypes(include=["number"])
# print out the correlation matrix
corr: pd.DataFrame = numeric.corr()
# printing only a slice of the matrix in the console
print(corr.iloc[:8, :8])

# Now we want to find the strongest correlation paris
pairs = (
    corr.where(~(corr == 1.0)) # replaces diagonal values with NaN so they don't become the highest correlations
    .abs()                     #  compare strength of corr regardless of sign
    .unstack()                 # convert 2D matrix to a 1D list of correlation pairs
    .dropna()                  # and drop empty values of NaNs
    .sort_values(ascending=False) # sort the values in descending order (largest values first)
)
# Now remove duplicate pairs of correlations
# 1) Loop through correlation pairs from strongest to weakest
# 2) Ignore duplicate reversed pairs
# 3) Keep adding unique pairs to a list
# 4) Stop once we have 15
seen = set()
top_pairs = []
for (a,b), v in pairs.items():
    if (b, a) in seen:
        continue
    seen.add((a, b))
    top_pairs.append((a, b, v))
    if len(top_pairs) == 15:
        break

print("\n=== EDA: TOP 15 STRONGEST CORRELATION PAIRS ===")
for a, b, v in top_pairs:
    print(f"{a:24s} vs {b:24s} | | corr| = {v:.3f}")

# Now for the final part of EDA, we compare which features relate most to winnings, rankings and wins
targets = ["Winnings, Rankings, Wins"]

for t in targets:
    if t in corr.columns:
        print(f"\n===EDA: TOP CORRELATIONS WITH {t} ===")
        print(corr[t].sort_values(ascending=False).head(12))


# And finally, examine these correlations as plots to show shape, outliers, and non-linearity
def scatter(x_col: str, y_col: str, alpha: float = 0.35):
    plt.figure(figsize=(8,5))
    plt.scatter(atp_stats[x_col], atp_stats[y_col], alpha = alpha)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.show()

# Plots a few appropriate relationships to start (swap these out as preferred)
print("\n=== EDA: PLOTTING SENSIBLE RELATIONSHIPS ===")
scatter("Wins", "Winnings")
scatter("Ranking", "Winnings")
scatter("Aces", "Wins")
scatter("DoubleFaults", "Winnings")
scatter("TotalPointsWon", "Winnings")

# -----------------------------------------------------------------------
## BEGIN LINEAR REGRESSION MODELLING
# Define a function that fits and evaluates target and feature variables
# Using the R-SQUARED to explain variance
# And using the RMSE: Average error size (in units of the target variable)
# X = features, y = target

def fit_and_eval(feature_cols, target_cols, test_size=0.2, random_state=42):
    """I split the data into training and testing data, fit a linear regression model
       and evaluate on both the training and test data.
       I do this on both sets because:
          - Training metrics give me an idea as to how well the model fits data its already seen
          - Test metrics tell me how well the model generalises to unseen data"""

    X = atp_stats[feature_cols]
    y = atp_stats[target_cols]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=random_state)

    # Build and fit the model on the training data
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)

    # predict on both the train and test set
    y_pred_train = mlr.predict(X_train)
    y_pred_test = mlr.predict(X_test)

    # Evaluate the model
    # RMSE (Root Mean Squared Error): average error size (in targets units)
    rmse_train = mean_squared_error(y_train, y_pred, squared=False)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    # R^2 (coefficient of determination) explains variance (answers closer to 1 are better)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    return (
        mlr,
        rmse_train, rmse_test,
        r2_train, r2_test,
        X_train, X_test,
        y_train, y_test,
        y_pred_train, y_pred_test,
    )

def plot_pred_vs_actual(y_test, y_pred_test, title="Predicted vs. Actual"):
    """I then define a function to plot predicted values against actual values for the test set.
    This tells me:
       - How close the predictions are to the truth overall
       - If points are roughly on the diagonal line
       - Whether the model systematically under/over-predicts at high values
    Each dot is one test example:
      - x-axis = actual value
      - y-axis = predicted value
      If predictions were perfect, all dots would lie on the diagonal line y = x. With the plot I can identify:
         - Big overall error - cloud of points far from diagonal
         - Nonlinear relationships (curve shapes)
         - If high-earners are consistently underpredicted"""

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred_test, alpha=0.4)

    #y = x reference line - where perfect predictions would lie
    min_v = min(float(y_test.min()), float(y_pred_test.min()))
    max_v = max(float(y_test.max()), float(y_pred_test.max()))
    plt.plot([min_v, max_v], [min_v, max_v])

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.show()


def plot_residuals(y_test, y_predictor, title="Residual analysis"):
    """I then carry out some residual analysis for the test set.

    This tells me:
       - If the errors are random (what linear regression assumes)
       - If the error size grows with my predictions (heteroscedasticity). This would create a 'fan shape'
       - If there is a curve pattern (non-linearity)
       - If residuals are skewed - representing outliers and non-normality

    Residual = Actual - Predicted.

    I create two standard plots:
    1) Residuals vs Predicted: this should look like random scatter around 0.
    2) Histogram of residuals: this should be roughly symmetric around 0.
    """
    residuals = y_test - y_predictor

    # Plot 1: residuals vs predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_predictor, residuals, alpha=0.4)
    plt.axhline(0)  # reference line at residual = 0
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(title + " (Residuals vs Predicted)")
    plt.show()

    # Plot 2: histogram of residuals
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title(title + " (Residual distribution)")
    plt.show()


def print_coefficients(mlr, feature_cols):
    """Print the fitted intercept and coefficients for explainability."""
    print("\nModel intercept:", mlr.intercept_)
    for name, coef in zip(feature_cols, mlr.coef_):
        print(f"{name:22s} coef = {coef:.6f}")

# ------------------------------------------------------------
## BUILD A SINGLE-FEATURE LINEAR REGRESSION MODEL

# We will choose winnings to be our target variable
target = "Winnings"
# Now we will choose a single independent variable - Aces
feature = "Aces"

print("\n=== SINGLE-FEATURE MODEL ===")
print(f"Model: {target} ~ {feature}")

(
    model,
    rmse_train, rmse_test,
    r2_train, r2_test,
    X_train, X_test,
    y_train, y_test,
    y_pred_train, y_pred_test,
) = fit_and_eval([feature], target)


print(f"R^2 on train set: {r2_train:.3f}")
print(f"R^2 on test set:  {r2_test:.3f}")
print(f"RMSE on train set: {rmse_train:,.0f}")
print(f"RMSE on test set:  {rmse_test:,.0f}")

# Show the fitted parameters (intercept and coefficient(s))
print_coefficients(model, [feature])

# Visual checks of performance on the TEST set
plot_pred_vs_actual(y_test, y_pred_test, title=f"Predicted vs Actual ({target} ~ {feature})")
plot_residuals(y_test, y_pred_test, title=f"{target} ~ {feature}")


# ------------------------------------------------------------
## COMPARING MULTIPLE SINGLE-FEATURE MODELS

# Pick a shortlist of candidate features
candidate_features = [
    "Wins",
    "Losses",
    "Ranking",
    "Aces",
    "DoubleFaults",
    "TotalPointsWon",
    "ServiceGamesWon",
    "ReturnGamesWon",
    "FirstServePointsWon",
    "SecondServePointsWon",
]

print("\n=== MULTIPLE SINGLE-FEATURE MODELS (predicting Winnings) ===")

single_results = []
for f in candidate_features:
    _, _, rmse_f, _, r2_f, *_ = fit_and_eval([f], target)
    single_results.append((f, r2_f, rmse_f))

# Sort by best R^2
single_results.sort(key=lambda t: t[1], reverse=True)

print("Best single-feature models (sorted by R^2):")
for f, r2_f, rmse_f in single_results:
    print(f"{target} ~ {f:22s} | R^2={r2_f:6.3f} | RMSE={rmse_f:10.0f}")

best_single_feature = single_results[0][0]
print("\nBest single feature (by R^2):", best_single_feature)
