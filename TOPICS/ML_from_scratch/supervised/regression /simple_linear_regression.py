"""
Machine Learning from Scratch: Simple Linear Regression
-------------------------------------------------------------------------
1. WHAT THIS SCRIPT DOES:
   This script trains a model to predict a continuous variable (Revenue) based on
   a single input variable (Month). It finds the "Line of Best Fit" (y = mx + b)
   by iteratively nudging the line until the error is minimized.

2. THE KEY DIFFERENCE:
   * Standard Workflow (sklearn):
       from sklearn.linear_model import LinearRegression
       model = LinearRegression()
       model.fit(X, y)  <-- Uses Ordinary Least Squares (OLS) (Matrix maths) behind the scenes.

   * My Approach:
       I manually implement Gradient Descent. I calculate the partial
       derivatives of the error function with respect to the slope (m) and
       intercept (b) to find the direction of steepest descent.

       I explicitly update weights in a loop:
       new_weight = old_weight - (learning_rate * gradient)

3. KEY MATH CONCEPTS:
   * Cost Function (MSE): Mean Squared Error
   * Gradient Descent: The optimisation algorithm used to minimize the Cost.
-------------------------------------------------------------------------
"""


import os
import matplotlib.pyplot as plt  # Only used for visualization, not logic!


class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialise the repressor with hyperparameters.

        Args:
            learning_rate (float): The step size for gradient descent (alpha).
                                   Too small = slow convergence.
                                   Too large = overshoot the minimum.
            iterations (int): How many times we loop through the dataset to update weights.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0.0  # Slope (weight)
        self.b = 0.0  # Intercept (bias)
        self.history = []  # To track how loss decreases over time

    def predict(self, X):
        """
        Generates predictions for a list of input values X.
        Formula: y = mx + b
        """
        return [(self.m * x + self.b) for x in X]

    def _calculate_gradients(self, X, y):
        """
        Calculates the gradients (derivatives) for m and b based on the current error.

        Maths:
        d_m = (-2/n) * Σ x * (y - (mx + b))
        d_b = (-2/n) * Σ (y - (mx + b))
        """
        n = len(X)

        # Initialise gradients to 0
        m_gradient = 0
        b_gradient = 0

        for i in range(n):
            # 1. Get the current actual values
            x_val = X[i]
            y_val = y[i]

            # 2. Make a prediction with current m and b
            prediction = self.m * x_val + self.b

            # 3. Calculate the difference (error)
            error = y_val - prediction

            # 4. Accumulate the gradients
            # The derivative with respect to m involves multiplying by x
            m_gradient += -(2 / n) * x_val * error

            # The derivative with respect to b is just the error term scaled
            b_gradient += -(2 / n) * error

        return m_gradient, b_gradient

    def _compute_loss(self, X, y):
        """
        Calculates Mean Squared Error (MSE).
        Used primarily for monitoring performance, not for training logic.
        """
        n = len(X)
        total_error = 0
        predictions = self.predict(X)

        for i in range(n):
            total_error += (y[i] - predictions[i]) ** 2

        return total_error / n

    def fit(self, X, y):
        """
        Trains the model by updating m and b over n iterations.
        """
        print(f"Starting Training: Initial Loss = {self._compute_loss(X, y):.4f}")

        for i in range(self.iterations):
            # 1. Calculate gradients based on all data points
            grad_m, grad_b = self._calculate_gradients(X, y)

            # 2. Update parameters (stepping down the gradient)
            # I subtract because the gradient points UP the slope, and it should go DOWN.
            self.m = self.m - (self.learning_rate * grad_m)
            self.b = self.b - (self.learning_rate * grad_b)

            # 3. Log the loss every 100 iterations to verify convergence
            if i % 100 == 0:
                current_loss = self._compute_loss(X, y)
                self.history.append(current_loss)
                print(f"Iteration {i}: Loss {current_loss:.4f} | m = {self.m:.3f}, b = {self.b:.3f}")

        print("Training Complete.")


# ==========================================
# DRIVER CODE (To run this script directly)
# ==========================================
if __name__ == "__main__":
    # 1. Prepare data
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

    # 2. Instantiate and train model
    regressor = SimpleLinearRegression(learning_rate=0.001, iterations=5000)
    regressor.fit(months, revenue)

    # 3. Generate predictions for plotting
    predicted_line = regressor.predict(months)

    # 4. Set up the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(months, revenue, color='blue', label='Actual Data')
    plt.plot(months, predicted_line, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Months')
    plt.ylabel('Revenue')
    plt.title('Simple Linear Regression Results')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # NEW CODE: SAVING THE FIGURE
    # ---------------------------------------------------------

    # A. dynamic path finding
    # "Which path?" -> .../supervised/regression/simple_linear.py
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up two levels to find the root -> .../ML_from_scratch/
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

    # B. Define where to save
    # Target: .../ML_from_scratch/figures/simp_lin_regr/
    save_dir = os.path.join(root_dir, 'figures', 'simp_lin_regr')

    # C. Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # D. Save the file
    save_path = os.path.join(save_dir, 'regression_result.png')
    plt.savefig(save_path)

    print(f"\n[Artifacts] Graph saved to: {save_path}")

    # ---------------------------------------------------------

    # 5. Show the plot (Must come after savefig)
    plt.show()