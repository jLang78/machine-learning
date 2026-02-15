"""
Machine Learning from Scratch: K-Means Clustering
-------------------------------------------------------------------------
1. WHAT THIS SCRIPT DOES:
   This script groups data into 'k' clusters without knowing the labels.
   It discovers structure by finding the geometric centers (centroids)
   of dense groups of data points.

2. THE KEY DIFFERENCE:
   * Standard Workflow (sklearn):
       from sklearn.cluster import KMeans
       kmeans = KMeans(n_clusters=3)
       kmeans.fit(X)

   * MY approach (from scratch):
       I manually implement the Expectation-Maximisation loop:
       1. Expectation (Assignment): Loop through every point and find the
          closest centroid.
       2. Maximisation (Update): Calculate the average (mean) position of
          points in each cluster and move the centroid there.
       3. I manually check for convergence (i.e did the centroids stop moving?).

3. KEY MATH CONCEPTS:
   * Centroids: The geometric center of a cluster.
   * Variance Minimisation: K-Means tries to minimize the distance within clusters.
-------------------------------------------------------------------------
"""

import random
import math
import os
import sys
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# SETUP: Import the custom data loader
# ---------------------------------------------------------
# 1. Get the directory where this script is located
#    Currently: .../ML_from_scratch/unsupervised/clustering
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to 'unsupervised'
parent_dir = os.path.dirname(current_dir)

# 3. Go up another level to 'ML_from_scratch' (The Project Root)
root_dir = os.path.dirname(parent_dir)

# 4. Add the root to Python's search path
sys.path.append(root_dir)

# 5. Now we can import from utils
try:
    from utils.data_loader import load_csv
except ImportError:
    print(f"CRITICAL ERROR: Could not import 'utils'.\nPython thinks the root is: {root_dir}")
    sys.exit(1)


class KMeans:
    def __init__(self, k=3, max_iterations=100):
        """
        Args:
            k (int): Number of clusters we want to find.
            max_iterations (int): Safety limit to prevent infinite loops.
        """
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []  # Will hold [x, y, ...] coordinates
        self.clusters = {}  # Keys: 0 to k-1, Values: List of data points

    def _euclidean_distance(self, point1, point2):
        """
        Calculates distance between two points (lists of floats).
        """
        distance = 0.0
        for i in range(len(point1)):
            distance += (point1[i] - point2[i]) ** 2
        return math.sqrt(distance)

    def _initialize_centroids(self, data):
        """
        Randomly picks 'k' points from the data to start as centroids.
        """
        self.centroids = random.sample(data, self.k)

    def _assign_clusters(self, data):
        """
        Step 2: Assign every point to the closest centroid.
        """
        # Reset clusters
        self.clusters = {i: [] for i in range(self.k)}

        for point in data:
            distances = []
            for centroid in self.centroids:
                distances.append(self._euclidean_distance(point, centroid))

            # Find index of the closest centroid
            closest_index = distances.index(min(distances))
            self.clusters[closest_index].append(point)

    def _update_centroids(self):
        """
        Step 3: Move centroids to the center (mean) of their cluster.
        Returns: True if centroids changed, False if they stayed the same (convergence).
        """
        old_centroids = list(self.centroids)  # Copy

        for i in range(self.k):
            points = self.clusters[i]

            if not points:
                continue  # Avoid division by zero if a cluster is empty

            # Calculate mean for each dimension
            new_centroid = []
            num_dimensions = len(points[0])

            for dim in range(num_dimensions):
                dim_values = [p[dim] for p in points]
                avg = sum(dim_values) / len(dim_values)
                new_centroid.append(avg)

            self.centroids[i] = new_centroid

        # Check if centroids moved significantly
        total_shift = 0
        for i in range(self.k):
            total_shift += self._euclidean_distance(old_centroids[i], self.centroids[i])

        return total_shift > 0.0001

    def fit(self, data):
        """
        Runs the K-Means algorithm.
        """
        print(f"Initializing {self.k} centroids...")
        self._initialize_centroids(data)

        for i in range(self.max_iterations):
            self._assign_clusters(data)
            changed = self._update_centroids()

            if not changed:
                print(f"Converged after {i + 1} iterations.")
                break
        else:
            print("Reached max iterations without full convergence.")


# ==========================================
# DRIVER CODE
# ==========================================
if __name__ == "__main__":

    # 1. Load Data
    print("Loading Data...")

    # --- CHECK YOUR FILENAME HERE ---
    # Ensure this matches 'Iris.csv' or 'iris.csv' exactly!
    X, y = load_csv('Iris.csv')

    if not X:
        print("Error: Dataset not found. Check path or filename.")
        sys.exit(1)

    # For K-Means, we only want the features (X), not the labels (y)
    # And we'll just use the first 2 columns (Sepal Length, Sepal Width) for easy 2D plotting
    X_2d = [row[:2] for row in X]

    # 2. Train Model
    k_value = 3
    kmeans = KMeans(k=k_value)
    kmeans.fit(X_2d)

    # 3. Visualization
    print("Plotting results...")
    plt.figure(figsize=(10, 6))

    # Define colors for the clusters
    colors = ['red', 'green', 'blue', 'cyan', 'magenta']

    # Plot each cluster
    for cluster_index in kmeans.clusters:
        points = kmeans.clusters[cluster_index]
        if not points: continue

        # Unzip points into x and y lists
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]

        plt.scatter(x_vals, y_vals, color=colors[cluster_index % len(colors)], s=30, alpha=0.6,
                    label=f'Cluster {cluster_index}')

    # Plot Centroids (Make them big and black)
    centroid_x = [c[0] for c in kmeans.centroids]
    centroid_y = [c[1] for c in kmeans.centroids]
    plt.scatter(centroid_x, centroid_y, color='black', marker='X', s=200, label='Centroids')

    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(f'K-Means Clustering from Scratch (k={k_value})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 4. Save Logic
    # We use 'root_dir' calculated at the top
    save_dir = os.path.join(root_dir, 'figures', 'kmeans')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'kmeans_result.png')

    plt.savefig(save_path)
    print(f"\n[Artifacts] Graph saved to: {save_path}")

    plt.show()

