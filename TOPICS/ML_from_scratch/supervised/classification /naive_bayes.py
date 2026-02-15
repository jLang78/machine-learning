"""
Machine Learning from Scratch: Gaussian Naive Bayes
----------------------------------------------------------------------
1. WHAT THIS SCRIPT DOES:
   I am building a probabilistic classifier based on Bayes' Theorem.
   Instead of looking at neighbors (like KNN), I look at the statistical
   distribution of the data. I calculate the mean and variance of each
   feature for each class, assuming they follow a Normal (Gaussian)
   distribution.

2. THE KEY DIFFERENCE:
   * Standard Workflow (sklearn):
       from sklearn.naive_bayes import GaussianNB
       model = GaussianNB()
       model.fit(X, y)

   * My Approach:
       I manually calculate the summary statistics (Mean, Variance) for
       each class. Then, for every prediction, I plug the new data into
       the Gaussian Probability Density Function (PDF) to see which class
       it most likely belongs to.

3. KEY MATH CONCEPTS:
   * Bayes' Theorem: P(A|B) = (P(B|A) * P(A)) / P(B)
   * Gaussian PDF: The "Bell Curve" formula used to calculate likelihoods.
   * Independence Assumption: I assume features don't affect each other.
-------------------------------------------------------------------------
"""

import math
import os
import sys
import matplotlib.pyplot as plt

# ---------------------------------------------------
# SETUP: Import my custom Data Loader
# -------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

# I insert the root at index 0 to ensure my local 'utils' is found first
sys.path.insert(0, root_dir)

from utils.data_loader import load_csv


class GaussianNaiveBayes:
    def __init__(self):
        # I will store the summary statistics (mean, variance) here.
        # Structure: { class_label: { feature_index: (mean, variance) } }
        self.summaries = {}
        # I also need to know the prior probability of each class (e.g., 33% Setosa)
        self.class_priors = {}

    def fit(self, X, y):
        """
        I am 'training' the model by calculating the Mean and Variance
        for every feature, separated by class.
        """
        n_samples = len(X)
        self.classes = list(set(y))

        # 1. Separate data by class
        separated = {c: [] for c in self.classes}
        for i in range(n_samples):
            separated[y[i]].append(X[i])

        # 2. Calculate statistics for each class
        for class_val, rows in separated.items():
            # Calculate Prior: P(Class) = count(Class) / total_samples
            self.class_priors[class_val] = len(rows) / n_samples

            # Calculate Mean and Variance for each column (feature)
            # I use zip(*rows) to transpose the matrix and get columns
            self.summaries[class_val] = []
            for column in zip(*rows):
                mean = sum(column) / len(column)
                # Variance = sum((x - mean)^2) / (n - 1)
                variance = sum([(x - mean) ** 2 for x in column]) / (len(column) - 1)
                self.summaries[class_val].append((mean, variance))

        print(f"I have learned statistics for {len(self.classes)} classes.")

    def _calculate_probability(self, x, mean, variance):
        """
        I calculate the Gaussian Probability Density Function (PDF).
        This tells me: "How likely is this value 'x' to occur in a distribution
        defined by this mean and variance?"
        """
        if variance == 0: return 0  # Avoid division by zero
        exponent = math.exp(-((x - mean) ** 2 / (2 * variance)))
        return (1 / (math.sqrt(2 * math.pi * variance))) * exponent

    def _predict_one(self, x):
        """
        I apply Bayes' Theorem to predict the class for a single row 'x'.
        P(Class|Data) proportional to P(Data|Class) * P(Class)
        """
        posteriors = {}

        for class_val in self.classes:
            # Start with the Prior Probability P(Class)
            prior = self.class_priors[class_val]

            # Calculate Likelihood P(Data|Class)
            # Since I assume features are independent (Naive), I multiply their probabilities.
            likelihood = 1
            class_summaries = self.summaries[class_val]

            for i, (mean, variance) in enumerate(class_summaries):
                feature_val = x[i]
                prob = self._calculate_probability(feature_val, mean, variance)
                likelihood *= prob

            # Posterior = Prior * Likelihood
            posteriors[class_val] = prior * likelihood

        # I return the class with the highest posterior probability
        # This sorts the dictionary by value and picks the biggest one
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        """
        I run the prediction loop for every row in the dataset X.
        """
        predictions = []
        for row in X:
            predictions.append(self._predict_one(row))
        return predictions


def accuracy_score(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


def train_test_split(X, y, test_size=0.2):
    # I shuffle and split the data manually
    data = list(zip(X, y))
    # random.seed(42) # Uncomment if I want reproducible results
    import random
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    train = data[:split_index]
    test = data[split_index:]
    return [r[0] for r in train], [r[0] for r in test], [r[1] for r in train], [r[1] for r in test]


# ==========================================
# DRIVER CODE
# =======================================
if __name__ == "__main__":

    # 1. Load Data
    print("I am loading the Iris Dataset...")
    # NOTE: Ensure the filename matches (Capital 'I' if needed!)
    X, y = load_csv('Iris.csv')

    if not X:
        print("Error: Dataset not found.")
        sys.exit(1)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"I split the data: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # 3. Train Model
    print("I am training the Gaussian Naive Bayes model...")
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)

    # 4. Predict
    print("I am making predictions on the test set...")
    predictions = nb.predict(X_test)

    # 5. Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print(f"I achieved an Accuracy of: {accuracy * 100:.2f}%")

    # --------------------------------------------------
    # VISUALIZATION & SAVING
    # -------------------------------------------------------

    plt.figure(figsize=(10, 6))

    # I map string labels to colors for plotting
    unique_labels = list(set(y))
    colors = ['red', 'green', 'blue']

    # I plot the test data, colored by my PREDICTIONS
    # This lets us visually check if the clusters make sense
    for i, label in enumerate(unique_labels):
        # I filter points that I PREDICTED as this label
        # (Using only the first 2 dimensions for the 2D plot)
        x_points = [X_test[j][0] for j in range(len(X_test)) if predictions[j] == label]
        y_points = [X_test[j][1] for j in range(len(X_test)) if predictions[j] == label]

        if x_points:
            plt.scatter(x_points, y_points, color=colors[i % len(colors)], label=f'Predicted {label}', alpha=0.7)

    # I also plot any incorrect predictions with a black 'x' to highlight errors
    errors_x = [X_test[j][0] for j in range(len(X_test)) if predictions[j] != y_test[j]]
    errors_y = [X_test[j][1] for j in range(len(X_test)) if predictions[j] != y_test[j]]
    if errors_x:
        plt.scatter(errors_x, errors_y, color='black', marker='x', s=100, label='Mistakes')

    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(f'Naive Bayes Predictions (Acc={accuracy:.2f})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save Logic
    save_dir = os.path.join(root_dir, 'figures', 'naive_bayes')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'nb_result.png')

    plt.savefig(save_path)
    print(f"\n[Artifacts] I saved the graph to: {save_path}")

    plt.show()