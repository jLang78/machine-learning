# Machine Learning from Scratch

As Feynman said - *"What I cannot create, I do not understand."*

## Project Purpose 
Most modern machine learning relies on powerful abstractions like Scikit-Learn, TensorFlow, or PyTorch. While these tools are essential for production, they often obscure the mathematical intuition and algorithmic logic that powers machine learning.

**This project looks at the 'how' of these models .**

The goal of this repository is **explainability**. I'm stripping away the imported libraries to build core ML algorithms using nothing but Python's standard library and raw mathematics. By manually implementing gradients, distance metrics, and optimisation loops, I gain a white-box understanding of how the "learning" actually happens.

## The Rules
To ensure deep understanding, this project adheres to strict constraints:
1.  **No Dataframes:** No `pandas`. Data is manipulated using lists of lists or dictionaries.
2.  **No Matrix Libraries:** No `numpy`. Dot products, transpositions, and matrix math are implemented from scratch.
3.  **No Pre-built Models:** No `sklearn`. E.g. If I want a Logistic Regression, I build the sigmoid function and the cost optimization loop myself
4.  **Visualisation:** `matplotlib` is permitted solely for visualising results, but not for data manipulation.

## Project Structure

### 1. Supervised Learning
Algorithms that learn a mapping from input variables ($X$) to output variables ($Y$) using labeled training data.

* **Regression:** Predicting continuous values (e.g., House Prices, Salary).
    * *Target Algorithms:* Simple Linear Regression, Multiple Linear Regression.
* **Classification:** Predicting discrete categories (e.g., Spam vs. Ham, Iris Species).
    * *Target Algorithms:* k-Nearest Neighbors (k-NN), Logistic Regression, Naive Bayes, Decision Trees.

### 2. Unsupervised Learning
Algorithms that model the underlying structure or distribution in the data ($X$) without reference to labeled outcomes.

* **Clustering:** Grouping similar data points together.
    * *Target Algorithms:* k-Means, Hierarchical Clustering.
* **Dimensionality Reduction:** Reducing the number of random variables under consideration.
    * *Target Algorithms:* Principal Component Analysis (PCA).
* **Density Estimation:** Estimating the probability density function of the random process.
    * *Target Algorithms:* Gaussian Mixture Models (GMM), Kernel Density Estimation (KDE).
* **Association:** Discovering interesting relations between variables in large databases.
    * *Target Algorithms:* Apriori Algorithm.
* **Anomaly Detection:** Identifying rare items, events, or observations.
    * *Target Algorithms:* Z-Score Outlier Detection, Isolation Forest (simplified).

## Datasets
For consistency, I utilise a few simple datasets across multiple algorithms to see how different approaches handle the same data.
* **Iris Dataset:** Used for Classification and Clustering.
* **Salary/Experience Dataset:** Used for Simple Linear Regression.

Please also note, that this is am iterative process, and more algorithms will be added as I come to terms with how they work 'under-the'hood' 
