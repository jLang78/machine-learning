# Credit Card Fraud Detection System 

## Project Overview
A machine learning pipeline to detect fraudulent credit card transactions. 
This project solves the problem of highly imbalanced data, by prioritising **Precision** (reducing false alarms) without sacrificing **Recall** (catching thieves).

* **Dataset**: 284,807 transactions (only 0.17% fraud).
* **Goal**: Flag fraudulent transactions while minimising customer annoyance (False Positives).

## Key Insights (EDA)
1.  **The Night-Time Pattern**: Fraudulent activity doesn't drop off during the night (3 AM - 5 AM) unlike normal human activity.
2.  **Fraud Amounts Higher**: Fraudsters typically make larger transactions (100-300 USD) compared to the average legitimate purchase (10-50 USD).
3.  **Imbalance**: A standard "Accuracy" metric is misleading. A model predicting "No Fraud" every time achieves 99.83% accuracy but fails the business objective.

## Model Performance
I experimented with Logistic Regression, Random Forest, and Gradient Boosting.

| Model                   | False Alarms (Lower is Better) | Frauds Caught (Higher is Better) | Verdict |
|:------------------------| :--- | :--- | :--- |
| **Logistic Regression** | 1,447 | 90 | Too aggressive (High cost). |
| **Gradient Boosting**   | 258 | 87 | Good, but noisy. |
| **Random Forest**       | **19** | **79** | **Best Balance.** |

**Best Model:** Random Forest
* **Precision:** ~80% (When the model flags a card, it is usually right).
* **Recall:** ~80% (It catches the vast majority of fraud).
* **Business Impact:** Reduces false declines by **99%** compared to the baseline Logistic Regression model.

## Tech Stack & Basic Project Structure
* **Python 3.13**
* **Scikit-Learn** (Modeling)
* **Pandas & NumPy** (Data Manipulation)
* **Seaborn & Matplotlib** (Visualisation)

```text
├── data/
│   ├── raw/            # Original dataset
│   └── processed/      # Cleaned test data (X_test.csv)
├── notebooks/          # Jupyter notebooks for EDA
├── models/             # Saved model binaries (model.joblib)
├── reports/            # Generated analysis
│   └── figures/        # Confusion Matrix & PR Curve
├── src/                # Source code
│   ├── data.py         # Data loading script
│   ├── features.py     # Preprocessing & Splitting
│   ├── model.py        # Model definitions
│   ├── train.py        # Training pipeline
│   └── evaluate.py     # Evaluation pipeline
└── README.md