"""
Machine Learning from Scratch: Decision Tree Classifier
-------------------------------------------------------------------------
1. WHAT THIS SCRIPT DOES:
   I am building a binary tree that learns to classify data by asking a
   sequence of Yes/No questions.

   I start with all data at the top (Root). I look for the "Best Split"
   (a feature and a threshold) that separates the classes most cleanly.
   I then recursively split the data until the groups are pure or I reach
   a maximum depth.

2. THE KEY DIFFERENCE:
   * Standard Workflow (sklearn):
       from sklearn.tree import DecisionTreeClassifier
       model = DecisionTreeClassifier()
       model.fit(X, y)

   * My Approach (from scratch):
       I manually implement "Recursive Partitioning".
       1. I write a function to calculate 'Gini Impurity' (how mixed a group is).
       2. I loop through EVERY feature and EVERY possible threshold to find
          the split that reduces Gini Impurity the most (Information Gain).
       3. I build a Node object for that split and repeat the process for
          the left and right branches.

3. KEY MATH CONCEPTS:
   * Gini Impurity: 1 - sum(probability_of_each_class^2)
   * Information Gain: Parent_Impurity - Weighted_Average_Child_Impurity
   * Recursion: A function that calls itself to build the tree.
-------------------------------------------------------------------------
"""

import os
import sys
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# SETUP: Import my custom Data Loader
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

# I insert the root at index 0 so my local 'utils' is found first
sys.path.insert(0, root_dir)

from utils.data_loader import load_csv


class Node:
    """
    I represent a single decision point in the tree.
    """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        # Decision Node attributes
        self.feature_index = feature_index  # Which column index (e.g., 2 for Petal Length)
        self.threshold = threshold  # The value to split on (e.g., < 2.5)
        self.left = left  # The Node to go to if True
        self.right = right  # The Node to go to if False

        # Leaf Node attributes
        self.value = value  # If I am a leaf, this is the predicted class


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def _calculate_gini(self, y):
        """
        I calculate the Gini Impurity of a list of labels.
        Formula: 1 - sum(p_i^2)
        """
        class_counts = {}
        for label in y:
            class_counts[label] = class_counts.get(label, 0) + 1

        impurity = 1
        for label in class_counts:
            prob = class_counts[label] / len(y)
            impurity -= prob ** 2

        return impurity

    def _split(self, X_column, split_thresh):
        """
        I split the data indices into left and right based on the threshold.
        """
        left_idxs = []
        right_idxs = []
        for i, val in enumerate(X_column):
            if val <= split_thresh:
                left_idxs.append(i)
            else:
                right_idxs.append(i)
        return left_idxs, right_idxs

    def _best_split(self, X, y):
        """
        I iterate through every feature and every threshold to find the
        split that gives the highest Information Gain.
        """
        best_gain = -1
        split_idx, split_thresh = None, None

        current_uncertainty = self._calculate_gini(y)
        n_features = len(X[0])

        for feature_index in range(n_features):
            # Get the column
            X_column = [row[feature_index] for row in X]
            unique_values = set(X_column)  # Only try unique values as thresholds

            for threshold in unique_values:
                left_idxs, right_idxs = self._split(X_column, threshold)

                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                # Calculate Information Gain
                n = len(y)
                n_l, n_r = len(left_idxs), len(right_idxs)
                gini_l = self._calculate_gini([y[i] for i in left_idxs])
                gini_r = self._calculate_gini([y[i] for i in right_idxs])

                child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r
                information_gain = current_uncertainty - child_gini

                if information_gain > best_gain:
                    best_gain = information_gain
                    split_idx = feature_index
                    split_thresh = threshold

        return split_idx, split_thresh

    def _most_common_label(self, y):
        """
        I find the majority class in a list of labels (Voting).
        """
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        # Sort by count descending
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return sorted_counts[0][0]

    def _build_tree(self, X, y, depth=0):
        """
        I recursively build the tree.
        """
        n_samples = len(X)
        n_labels = len(set(y))

        # Stopping Criteria (Leaf Node)
        # 1. Reached max depth
        # 2. Only 1 class left (pure)
        # 3. Too few samples to split
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        feature_idx, threshold = self._best_split(X, y)

        # If no split improves gain, become a leaf
        if feature_idx is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Create Child Nodes
        # Extract data for left and right
        left_idxs, right_idxs = self._split([row[feature_idx] for row in X], threshold)

        left_X = [X[i] for i in left_idxs]
        left_y = [y[i] for i in left_idxs]
        right_X = [X[i] for i in right_idxs]
        right_y = [y[i] for i in right_idxs]

        # Recursion!
        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)

        return Node(feature_index=feature_idx, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        print("I am growing the decision tree...")
        self.root = self._build_tree(X, y)
        print("I have finished growing the tree.")

    def _traverse_tree(self, x, node):
        """
        I walk down the tree for a single data point 'x'.
        """
        if node.value is not None:
            return node.value  # Found a leaf!

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return predictions

    def print_tree(self, node=None, indent="  "):
        """
        I print the tree structure to the console so you can see the logic.
        """
        if node is None:
            node = self.root

        if node.value is not None:
            print(f"{indent}Predicted Class: {node.value}")
        else:
            print(f"{indent}Is Feature {node.feature_index} <= {node.threshold}?")
            print(f"{indent}True ->")
            self.print_tree(node.left, indent + "    ")
            print(f"{indent}False ->")
            self.print_tree(node.right, indent + "    ")


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def accuracy_score(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]: correct += 1
    return correct / len(y_true)


def train_test_split(X, y, test_size=0.2):
    import random
    data = list(zip(X, y))
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_size))
    train = data[:split_idx]
    test = data[split_idx:]
    return [r[0] for r in train], [r[0] for r in test], [r[1] for r in train], [r[1] for r in test]


# ==========================================
# DRIVER CODE
# ==========================================
if __name__ == "__main__":

    # 1. Load Data
    print("I am loading the Iris Dataset...")
    X, y = load_csv('Iris.csv')  # Ensure case matches file!

    if not X:
        print("Error: Dataset not found.")
        sys.exit(1)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"I split the data: {len(X_train)} training, {len(X_test)} testing.")

    # 3. Train Model
    clf = DecisionTree(max_depth=5)
    clf.fit(X_train, y_train)

    # Print the logic
    print("\n--- The Tree I Grew ---")
    clf.print_tree()
    print("-----------------------\n")

    # 4. Predict
    predictions = clf.predict(X_test)

    # 5. Evaluate
    acc = accuracy_score(y_test, predictions)
    print(f"I achieved an Accuracy of: {acc * 100:.2f}%")

    # ---------------------------------------------------------
    # VISUALIZATION & SAVING
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    unique_labels = list(set(y))
    colors = ['red', 'green', 'blue']

    # Plot predictions
    for i, label in enumerate(unique_labels):
        x_points = [X_test[j][0] for j in range(len(X_test)) if predictions[j] == label]
        y_points = [X_test[j][1] for j in range(len(X_test)) if predictions[j] == label]

        if x_points:
            plt.scatter(x_points, y_points, color=colors[i % len(colors)], label=f'Predicted {label}')

    # Mark errors
    errors_x = [X_test[j][0] for j in range(len(X_test)) if predictions[j] != y_test[j]]
    errors_y = [X_test[j][1] for j in range(len(X_test)) if predictions[j] != y_test[j]]
    if errors_x:
        plt.scatter(errors_x, errors_y, color='black', marker='x', s=100, label='Mistakes')

    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(f'Decision Tree Predictions (Acc={acc:.2f})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save Logic
    save_dir = os.path.join(root_dir, 'figures', 'decision_tree')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'dt_result.png')

    plt.savefig(save_path)
    print(f"\n[Artifacts] I saved the graph to: {save_path}")

    plt.show()