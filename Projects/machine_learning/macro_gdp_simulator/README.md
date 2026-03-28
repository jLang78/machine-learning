# Macroeconomic Policy Simulator: Drivers of Global Wealth

## 1. Project Purpose
I developed this project to simulate how structural economic changes and policy interventions impact a nation's wealth. Rather than simply predicting a static number, this tool explains the causal relationships behind macroeconomic growth, making it highly relevant for economic consulting and policy analysis. 

I pull live, real-world data directly from the World Bank API to evaluate how human capital (health, labor workforce), physical capital (investment), and macroeconomic stability (inflation, trade openness) drive Gross Domestic Product (GDP) per capita. The final output is an interactive Streamlit dashboard designed for non-technical stakeholders to simulate policy changes, adjust economic levers, and instantly visualize the projected economic outcomes.

## 2. Machine Learning & Econometric Approach
To achieve this, I utilise a Multiple Linear Regression model with targeted polynomial feature engineering. I deliberately prioritize interpretability and causal inference over black-box predictive algorithms.

* **Log-Linear Transformation:** Because national wealth compounds exponentially rather than linearly, I apply a natural logarithm to the target variable. The core model takes the form:
  
  $$\ln(\text{GDP per Capita}) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \epsilon$$

  This allows the model to interpret coefficients as percentage changes, providing clear business value. For example, the model mathematically proves that a 1 unit increase in labor participation correlates to a specific percentage rise in GDP.
  
* **Polynomial Feature Engineering:** Through rigorous residual analysis, I identified a non-linear, U-shaped error distribution in the Life Expectancy variable. I engineered a squared polynomial term to capture this specific curve:

  $$Y = \beta_1(\text{Life Expectancy}) + \beta_2(\text{Life Expectancy})^2$$

  

  This mathematically proves the economic theory of diminishing returns. The initial economic boom of improving a developing nation's health is massive, but it plateaus and shifts as the population ages.

* **Algorithm Selection:** I execute the model using the `statsmodels` library (Ordinary Least Squares) rather than `scikit-learn`. This ensures I have access to strict econometric diagnostic metrics, including standard errors, t-statistics, p-values, and condition numbers.

## 3. Architecture & File Structure
I structure this project as a reproducible, automated data pipeline. By keeping the heavy data files ignored via the `.gitignore` file, anyone can clone this repository and reconstruct the analysis from scratch.

```text
data/
├── raw/                            <-- Stores the raw API pulls from the World Bank.
└── processed/                      <-- Stores the cleaned, imputed, and scaled matrices ready for modeling.

notebooks/
├── 01_data_extraction.ipynb        <-- I ingest the API data and analyze missingness using visual matrices.
├── 02_linear_regression.ipynb      <-- I run the baseline OLS model and analyze residual errors.
└── 03_poly_transformation.ipynb    <-- I mathematically apply transformations to fix non-linearities.

models/
└── gdp_regression_model.pkl        <-- The serialized, trained mathematical model.

src/
├── data.py                         <-- Automated script to dynamically query the World Bank API.
├── features.py                     <-- Automated script to apply strict numeric coercion and polynomial math.
├── model.py                        <-- Automated script to train the OLS model and implement failsafes.
└── dashboard.py                    <-- The interactive Streamlit and Plotly web application.
```

## 4. User Instructions
To run this pipeline locally and launch the interactive dashboard, I follow these steps:

**Step 1: Install Dependencies**
I ensure Python is installed, then I install the required packages:
```bash
pip install -r requirements.txt
```

**Step 2: Execute the Automated Pipeline**
I built the backend to sequentially fetch live data, process the mathematics, and train the model. I run these commands in order from my terminal:
```bash
python -m src.data
python -m src.features
python -m src.model
```

**Step 3: Launch the Dashboard**
Once the model is trained and saved, I launch the interactive user interface:
```bash
streamlit run src/dashboard.py
```

## 5. Evaluation Metrics & Future Improvements



**Model Performance:**
The final polynomial model achieves an R-squared value of $R^2 = 0.784$. This means my selected features successfully explain 78.4% of the variance in global wealth across 257 regions. Furthermore, the residual plots confirm that the polynomial transformation successfully neutralizes the non-linear errors previously found in the human capital metrics.

**Econometric Diagnostics & Limitations:**
* **Multicollinearity:** The model summary reported a high condition number ($1.81 \times 10^3$). Macroeconomic variables are inherently highly correlated (e.g., countries with high trade openness also tend to have high foreign direct investment). This can inflate standard errors. 

**Future Improvements:**
* **Regularization:** Implementing a Ridge Regression (L2 penalization) could help stabilize the coefficients and mitigate the multicollinearity identified by the high condition number.
* **Panel Data Analysis:** Currently, the model looks at a cross-sectional snapshot of a single stable year (2022). Upgrading this to a Fixed Effects Panel Model that tracks countries across 20 years would isolate specific national idiosyncrasies, control for unobserved variables, and provide an even stronger proof of economic causality.