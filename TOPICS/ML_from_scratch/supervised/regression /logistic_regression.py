"""
Machine Learning from Scratch: Logistic Regression
-------------------------------------------------------------------------
1. WHAT THIS SCRIPT DOES:
   I am building a model to predict the probability of a binary outcome.
   (e.g., Is this flower 'Setosa' or 'Versicolor'?)

   IMPORTANT: Logistic regression is binary classifier, and the IRIS dataset has three classes.
              I therefore take steps to remove the third class.

   I use the linear equation (z = wx + b) combined with the Sigmoid
   activation function to map predictions to a probability between 0 and 1.

2. THE KEY DIFFERENCE:
   * Standard Workflow (sklearn):
       from sklearn.linear_model import LogisticRegression
       model = LogisticRegression()
       model.fit(X, y)

   * MY APPROACH :
       I manually implement the "Sigmoid" function.
       I manually implement the "Log Loss" (Binary Cross Entropy) cost function.
       I use Gradient Descent to optimize the weights, just like in Linear Regression,
       but using the probability error instead of raw distance.

3. KEY MATH CONCEPTS:
   * Sigmoid Function: 1 / (1 + e^-z) -> Maps any number to (0, 1).
   * Decision Boundary: The line where probability = 0.5 (where z = 0).
   * Log Loss: The penalty for being confident and wrong.
-------------------------------------------------------------------------
"""

import math
import os
import sys
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# SETUP: Import custom Data Loader
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

# I insert the root at index 0 so my local 'utils' is found first
sys.path.insert(0, root_dir)

from utils.data_loader import load_csv


class LogisticRegression:
    def __init__(self, learning_rate=0.1, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = []
        self.bias = 0
        self.loss_history = []

    def _sigmoid(self, z):
        """
        I squash the input 'z' into a probability between 0 and 1.
        I use 'max' to avoid overflow errors with very large negative numbers.
        """
        if z < -700: return 0  # Avoid python math range error
        return 1 / (1 + math.exp(-z))

    def _dot_product(self, row, weights):
        """
        I calculate the dot product of a data row and the weights.
        z = w1*x1 + w2*x2 ... + b
        """
        z = 0
        for i in range(len(row)):
            z += row[i] * weights[i]
        return z + self.bias

    def fit(self, X, y):
        """
        I train the model using Gradient Descent.
        """
        n_samples = len(X)
        n_features = len(X[0])

        # 1. Initialize weights to zero
        self.weights = [0.0] * n_features
        self.bias = 0.0

        print(f"I am training on {n_samples} samples with {n_features} features.")

        # 2. Gradient Descent Loop
        for i in range(self.iterations):

            # Reset gradients for this iteration
            dw = [0.0] * n_features
            db = 0.0
            total_loss = 0

            # Loop through every sample (Batch Gradient Descent)
            for j in range(n_samples):
                # A. Linear Step
                linear_pred = self._dot_product(X[j], self.weights)

                # B. Activation Step (Sigmoid)
                y_predicted = self._sigmoid(linear_pred)

                # C. Calculate Difference (Error term)
                error = y_predicted - y[j]

                # D. Accumulate Gradients
                # dw = sum(error * x)
                for f in range(n_features):
                    dw[f] += error * X[j][f]
                db += error

                # E. (Optional) Calculate Loss for monitoring
                # Log Loss: -[y*log(y_hat) + (1-y)*log(1-y_hat)]
                epsilon = 1e-15  # prevent log(0)
                y_pred_clipped = max(epsilon, min(1 - epsilon, y_predicted))
                total_loss += -(y[j] * math.log(y_pred_clipped) + (1 - y[j]) * math.log(1 - y_pred_clipped))

            # 3. Average the gradients
            dw = [grad / n_samples for grad in dw]
            db = db / n_samples

            # 4. Update Weights
            for f in range(n_features):
                self.weights[f] -= self.learning_rate * dw[f]
            self.bias -= self.learning_rate * db

            # Log progress
            if i % 100 == 0:
                avg_loss = total_loss / n_samples
                self.loss_history.append(avg_loss)
                print(f"Iteration {i}: Loss = {avg_loss:.4f}")

    def predict_proba(self, X):
        """
        I return the raw probability (0.0 to 1.0) for each sample.
        """
        predictions = []
        for row in X:
            z = self._dot_product(row, self.weights)
            prob = self._sigmoid(z)
            predictions.append(prob)
        return predictions

    def predict(self, X, threshold=0.5):
        """
        I convert probabilities to class labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probabilities]


# ==========================================
# DRIVER CODE

if __name__ == "__main__":

    # 1. Load Data
    print("Loading the Iris Dataset...")
    X_raw, y_raw = load_csv('Iris.csv')  # Check capitalization!

    if not X_raw:
        print("Error: Dataset not found.")
        sys.exit(1)

    # 2. Preprocessing: Binary Classification Only
    # I am going to keep only 'Iris-setosa' (0) and 'Iris-versicolor' (1)
    # I will ignore 'Iris-virginica' to make this a simple Yes/No problem.
    X = []
    y = []

    for i in range(len(y_raw)):
        if y_raw[i] == 'Iris-setosa':
            X.append(X_raw[i][:2])  # Keep only first 2 features (Sepal Length/Width) for plotting
            y.append(0)
        elif y_raw[i] == 'Iris-versicolor':
            X.append(X_raw[i][:2])
            y.append(1)

    print(f"Filtered the dataset to {len(X)} binary samples (Setosa vs Versicolor).")

    # 3. Train
    model = LogisticRegression(learning_rate=0.1, iterations=2000)
    model.fit(X, y)

    # 4. Visualization
    # To plot the "Decision Boundary" line, we solve for x2 where probability is 0.5
    # The equation is: w1*x1 + w2*x2 + b = 0
    # Therefore: x2 = -(w1*x1 + b) / w2

    plt.figure(figsize=(10, 6))

    # Plot data points
    setosa_x = [X[i][0] for i in range(len(X)) if y[i] == 0]
    setosa_y = [X[i][1] for i in range(len(X)) if y[i] == 0]
    versicolor_x = [X[i][0] for i in range(len(X)) if y[i] == 1]
    versicolor_y = [X[i][1] for i in range(len(X)) if y[i] == 1]

    plt.scatter(setosa_x, setosa_y, color='red', label='Iris-setosa (0)')
    plt.scatter(versicolor_x, versicolor_y, color='blue', label='Iris-versicolor (1)')

    # Plot Decision Boundary Line
    w1, w2 = model.weights
    b = model.bias

    # Generate x-values for the line
    x_min = min([row[0] for row in X])
    x_max = max([row[0] for row in X])

    # Calculate corresponding y-values
    # x2 = -(w1*x1 + b) / w2
    y_min = -(w1 * x_min + b) / w2
    y_max = -(w1 * x_max + b) / w2

    plt.plot([x_min, x_max], [y_min, y_max], color='green', linewidth=2, linestyle='--', label='Decision Boundary')

    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Logistic Regression: Decision Boundary')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save Logic
    save_dir = os.path.join(root_dir, 'figures', 'logistic_regression')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'log_reg_result.png')

    plt.savefig(save_path)
    print(f"\n[Artifacts] Saved the graph to: {save_path}")

    plt.show()