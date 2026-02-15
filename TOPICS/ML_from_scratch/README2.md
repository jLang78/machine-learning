# The Mathematical Recipes
*A linear, step-by-step guide to the math behind the code.*

This document removes the Python classes and focuses purely on the algorithm procedure

---

## 1. Simple Linear Regression (Gradient Descent)
**Goal:** Fit a straight line $y = mx + b$ through data points to minimize error.

### The Process
**Step 1: Initialisation**
* Start with random guesses for the slope ($m$) and intercept ($b$).
    * $m = 0$
    * $b = 0$

**Step 2: Prediction (The Forward Pass)**
* For every data point $x_i$, calculate the predicted value $\hat{y}_i$:
    $$\hat{y}_i = m \cdot x_i + b$$

**Step 3: Calculate Error (Cost Function)**
* Calculate how wrong the model is on average using Mean Squared Error (MSE):
    $$J(m, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Step 4: Gradient Calculation (The Derivative)**
* Find out which way is "downhill" for both $m$ and $b$.
* **Gradient for Slope ($m$):** (Average of $x \cdot error$)
    $$\frac{\partial J}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)$$
* **Gradient for Intercept ($b$):** (Average of error)
    $$\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

**Step 5: Update Weights (The Step)**
* Adjust $m$ and $b$ in the opposite direction of the gradient.
    $$m_{new} = m_{old} - (\alpha \cdot \frac{\partial J}{\partial m})$$
    $$b_{new} = b_{old} - (\alpha \cdot \frac{\partial J}{\partial b})$$
    *(Where $\alpha$ is the Learning Rate, e.g., 0.01)*

**Step 6: Repeat**
* Repeat Steps 2-5 for a fixed number of iterations (e.g., 1000 times) or until the error stops changing.

---

## 2. K-Nearest Neighbors (Classification)
**Goal:** Classify a new point based on the majority class of its closest neighbors.

### The Process
**Step 1: Store Data**
* No math here. Just hold onto the training data $(X, y)$.

**Step 2: Calculate Distances (Prediction Time)**
* A new point $p$ arrives. Calculate the distance between $p$ and **every single point** $q$ in the training set.
* **Euclidean Distance Formula:**
    $$d(p, q) = \sqrt{\sum_{i=1}^{k} (p_i - q_i)^2}$$
    *(In 2D: $\sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$)*

**Step 3: Sort & Select**
* Sort the list of distances from smallest to largest.
* Pick the top $k$ points (e.g., top 3).

**Step 4: Vote**
* Look at the classes (labels) of those $k$ points.
    * Neighbor 1: "Setosa"
    * Neighbor 2: "Setosa"
    * Neighbor 3: "Versicolor"
* **Result:** Majority wins ("Setosa").

---

## 3. K-Means Clustering (Unsupervised)
**Goal:** Group data into $k$ clusters by finding their geometric centers.

### The Process
**Step 1: Initialize Centroids**
* Pick $k$ random points from the dataset to be the initial "centers" ($\mu_1, \mu_2, ... \mu_k$).

**Step 2: Assignment (The "E" Step)**
* For every data point $x_i$:
    * Calculate distance to Centroid 1, Centroid 2, etc.
    * Assign $x_i$ to the cluster of the closest centroid.
    $$c^{(i)} := \text{argmin}_j ||x^{(i)} - \mu_j||^2$$

**Step 3: Update Centroids (The "M" Step)**
* For each cluster $j$:
    * Find the new center by taking the **average** (mean) of all points currently assigned to it.
    $$\mu_j = \frac{1}{|C_j|} \sum_{x \in C_j} x$$

**Step 4: Convergence Check**
* Did the centroids move?
    * **Yes:** Repeat Steps 2 & 3.
    * **No:** Stop. The clusters are stable.

---

## 4. Naive Bayes (Probabilistic)
**Goal:** Calculate the probability of a class given the data: $P(Class | Data)$.

### The Process
**Step 1: Training (Summarisation)**
* For each class $c$ (e.g., "Setosa"), calculate:
    * **Prior Probability:** How common is this class?
        $$P(c) = \frac{\text{Count}(c)}{\text{Total Samples}}$$
    * **Mean ($\mu$) and Variance ($\sigma^2$)** for every feature (e.g., Sepal Length).

**Step 2: Likelihood Calculation (Prediction Time)**
* For a new data point $x$ with feature value $v$, calculate the likelihood that $v$ came from class $c$ using the **Gaussian PDF**:
    $$P(x_i | c) = \frac{1}{\sqrt{2\pi\sigma_c^2}} e^{ -\frac{(x_i - \mu_c)^2}{2\sigma_c^2} }$$

**Step 3: Apply Bayes Theorem**
* Calculate the unnormalized probability (Posterior) for each class:
    $$\text{Posterior}(c) = P(c) \times \prod_{i=1}^{n} P(x_i | c)$$
    *(Prior $\times$ Product of all Feature Likelihoods)*

**Step 4: Select**
* Pick the class with the highest Posterior score.

---

## 5. Decision Trees (CART)
**Goal:** Split data recursively to create pure leaf nodes.

### The Process
**Step 1: Calculate Uncertainty (Gini Impurity)**
* Measure how "mixed" a group of data is.
    $$Gini = 1 - \sum_{i=1}^{C} (p_i)^2$$
    *(Where $p_i$ is the probability of class $i$ in that node)*.

**Step 2: Evaluate Splits**
* For every feature and every possible threshold value:
    1.  Split the data into Left Group and Right Group.
    2.  Calculate Gini for Left and Right.
    3.  Calculate **Weighted Gini** of the split:
        $$Gini_{split} = \frac{n_{left}}{n_{total}} Gini_{left} + \frac{n_{right}}{n_{total}} Gini_{right}$$

**Step 3: Calculate Information Gain**
* How much did this split help?
    $$Gain = Gini_{parent} - Gini_{split}$$

**Step 4: Recurse**
* Pick the split with the **Highest Gain**.
* Create a Node.
* Repeat Step 1 for the Left data and the Right data separately.
* **Stop when:** Max depth reached OR node is pure (Gini = 0).

## 6. Logistic Regression (Binary Classification)
**Goal:** Predict the probability that an input belongs to a specific class (0 or 1).

### The Process
**Step 1: The Linear Step (The Logits)**
* Just like Linear Regression, calculate the weighted sum of inputs.
    $$z = w \cdot x + b$$

**Step 2: The Activation (The Sigmoid)**
* "Squash" the linear result $z$ into a range between 0 and 1 using the Sigmoid function.
* This turns a raw number (like 5.2 or -13.0) into a probability (like 0.99 or 0.001).
    $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

**Step 3: Calculate Error (Log Loss)**
* I cannot use Mean Squared Error here because the Sigmoid makes the curve wavy (non-convex), which confuses Gradient Descent.
* Instead, I use **Log Loss** (Binary Cross-Entropy):
    * If actual $y=1$, I want $\hat{y}$ close to 1 (Error = $-\log(\hat{y})$).
    * If actual $y=0$, I want $\hat{y}$ close to 0 (Error = $-\log(1 - \hat{y})$).
    $$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log(\hat{y}_i) + (1-y_i) \log(1 - \hat{y}_i)]$$

**Step 4: Gradient Descent**
* Surprisingly, the derivative of the Log Loss function simplifies to the exact same formula as Linear Regression
* **Gradient for Weights ($w$):**
    $$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} x_i (\hat{y}_i - y_i)$$
* **Gradient for Bias ($b$):**
    $$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)$$

**Step 5: Update**
* Nudge the weights to minimize error.
    $$w_{new} = w_{old} - \alpha \cdot \frac{\partial J}{\partial w}$$