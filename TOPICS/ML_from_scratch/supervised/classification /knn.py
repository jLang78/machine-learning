"""
Machine Learning from Scratch: K-Nearest Neighbors (KNN)
--------------------------------------------------------
1. WHAT THIS SCRIPT DOES:
   This script classifies data points (Iris species) based on similarity.
   When given a new flower, it looks at the 'k' closest flowers in the
   training data and assigns the most common species among them.

2. THE KEY DIFFERENCE:
   * Standard Workflow (sklearn):
       from sklearn.neighbors import KNeighborsClassifier
       clf = KNeighborsClassifier(n_neighbors=3)
       clf.fit(X, y)
       # Sklearn optimises speed using complex data structures like KD-Trees.

   * My approach (from scratch):
       I implement a distance calculation.
       1. Write a function for Euclidean Distance: sqrt(sum((a-b)^2))
       2. For every prediction, calculate the distance to EVERY single
          training point (computationally expensive but easy to understand).
       3. Manually sort the distances and take the top 'k'.

3. KEY MATH CONCEPTS:
   * Euclidean Distance (L2 Norm): Geometry in n-dimensional space.
   * Voting Systems: How to resolve ties in clustering.
"""

import math
import operator
import random
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# SETUP: Import our custom Data Loader

import sys
import os

# Get the directory where this script is located: .../supervised/clustering
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level: .../supervised
parent_dir = os.path.dirname(current_dir)

# Go up another level: .../ML_from_scratch
root_dir = os.path.dirname(parent_dir)

# Adding root to Python's search path
sys.path.append(root_dir)

# import from utils
try:
    from utils.data_loader import load_csv
except ImportError:
    print(f"CRITICAL ERROR: Could not import 'utils'.\nPython thinks the root is: {root_dir}\nCheck directory structure!")
    sys.exit(1)

# ==================================================================
# OBJECT-ORIENTED CODE


class KNearestNeighbors:
    def __init__(self, k=3):


        # Args:
        #     k (int): The number of neighbors to consider.
        #              Usually an odd number to avoid ties in voting.

        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        """
        KNN is a 'Lazy Learner'. It doesn't actually 'learn' a model
        (such as finding coefficients in regression).
        It simply memorises the training data.
        """
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, point1, point2):
        """
        Calculates the square root of the sum of squared differences.
        Assumes point1 and point2 are lists of equal length.
        """
        distance = 0.0
        # Iterate over each feature (dimension)
        for i in range(len(point1)):
            distance += (point1[i] - point2[i]) ** 2
        return math.sqrt(distance)

    def _get_neighbors(self, test_point):
        """
        Finds the 'k' nearest neighbors in the training set for a given test point.
        """
        distances = []

        # 1. Calculate distance from test_point to every point in X_train
        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(test_point, self.X_train[i])
            # Store tuple: (distance, label)
            distances.append((dist, self.y_train[i]))

        # 2. Sort the list by distance (smallest to largest)
        distances.sort(key=operator.itemgetter(0))

        # 3. Pick the top 'k' neighbors
        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i])

        return neighbors

    def predict(self, X_test):
        """
        Predicts the class labels for a list of test data points.
        """
        predictions = []

        for test_point in X_test:
            # 1. Find neighbors
            neighbors = self._get_neighbors(test_point)

            # 2. Extract the labels of those neighbors
            # neighbor structure is (distance, label)
            neighbor_labels = [neighbor[1] for neighbor in neighbors]

            # 3. Vote (find the most common label)
            # I use a simple dictionary to count votes
            vote_counts = {}
            for label in neighbor_labels:
                if label in vote_counts:
                    vote_counts[label] += 1
                else:
                    vote_counts[label] = 1

            # Sort votes descending and pick the winner
            sorted_votes = sorted(vote_counts.items(), key=operator.itemgetter(1), reverse=True)
            winner = sorted_votes[0][0]
            predictions.append(winner)

        return predictions


def accuracy_score(y_true, y_pred):
    """
    Helper to calculate accuracy (Correct Predictions / Total Predictions)
    """
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


def train_test_split(X, y, test_size=0.2):
    """
    Splits data into training and testing sets randomly.
    """
    # Combine X and y so we can shuffle them together
    data = list(zip(X, y))
    random.shuffle(data)

    split_index = int(len(data) * (1 - test_size))

    train_data = data[:split_index]
    test_data = data[split_index:]

    # Unzip them back apart
    X_train = [row[0] for row in train_data]
    y_train = [row[1] for row in train_data]
    X_test = [row[0] for row in test_data]
    y_test = [row[1] for row in test_data]

    return X_train, X_test, y_train, y_test


# ==========================================
# DRIVER CODE

if __name__ == "__main__":

    # 1. Load Data
    print("Loading Iris Dataset...")
    X, y = load_csv('Iris.csv')

    # Handle case where file isn't found
    if not X:
        print("Please download the Iris dataset to the data/ folder first.")
        exit()

    # 2. Split Data
    # I train on 80%, test on 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

    # 3. Train Model
    k_value = 3
    print(f"Initialising KNN with k={k_value}...")
    knn = KNearestNeighbors(k=k_value)
    knn.fit(X_train, y_train)

    # 4. Predict
    print("Predicting...")
    predictions = knn.predict(X_test)

    # 5. Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # ---------------------------------------------------------
    # VISUALISATION & SAVING

    # I plot the first two features (Sepal Length vs Sepal Width)
    # and color them by their TRUE species to see the distribution.

    plt.figure(figsize=(10, 6))

    # Map string labels to colors for plotting
    # (I assume labels are strings like 'Iris-setosa')
    unique_labels = list(set(y))
    colors = ['red', 'green', 'blue']

    for i, label in enumerate(unique_labels):
        # Filter points that belong to this label
        # (This is a manual way to do boolean indexing without numpy)
        x_points = [X[j][0] for j in range(len(X)) if y[j] == label]  # Sepal Length
        y_points = [X[j][1] for j in range(len(X)) if y[j] == label]  # Sepal Width

        plt.scatter(x_points, y_points, color=colors[i % len(colors)], label=label)

    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(f'Iris Dataset Classification (k={k_value}, Acc={accuracy:.2f})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save Logic
    save_dir = os.path.join(root_dir, 'figures', 'knn_classification')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'knn_result.png')

    plt.savefig(save_path)
    print(f"\n[Artifacts] Graph saved to: {save_path}")

    plt.show()