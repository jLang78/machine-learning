
# src/dashboard.py
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import plotly.express as px
import plotly.graph_objects as go

# I capture the current working directory as the project root
# because Streamlit occasionally drops the __file__ variable
project_root = os.getcwd()

if project_root not in sys.path:
    sys.path.insert(0, project_root)

st.set_page_config(page_title="Macroeconomic Policy Simulator", layout="wide")

st.title("Macroeconomic Policy Simulator")
st.markdown(
    "I built this dashboard to simulate how structural economic changes impact a nation's wealth. I adjust the policy levers below to see the predicted effect on GDP per capita, and I explore the charts to understand the underlying econometric model.")


@st.cache_resource
def load_model():
    """
    I load the pre-trained OLS regression model.
    """
    model_path = os.path.join(project_root, "models", "gdp_regression_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_data():
    """
    I load the processed historical data to visualize actual vs. predicted accuracy.
    """
    data_path = os.path.join(project_root, "data", "processed", "world_bank_processed.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


model = load_model()
df = load_data()

if model is not None and df is not None:
    # -- SIDEBAR: USER INPUTS ---
    st.sidebar.header("Economic Policy Levers")

    trade = st.sidebar.slider("Trade Openness (% of GDP)", 10.0, 200.0, 50.0)
    fdi = st.sidebar.slider("Foreign Direct Investment (% of GDP)", -5.0, 40.0, 2.0)
    capital = st.sidebar.slider("Gross Capital Formation (% of GDP)", 10.0, 50.0, 25.0)
    life_exp = st.sidebar.slider("Life Expectancy (Years)", 50.0, 85.0, 70.0)
    labor = st.sidebar.slider("Labor Force Participation (%)", 40.0, 90.0, 60.0)
    pop_growth = st.sidebar.slider("Population Growth (Annual %)", -2.0, 5.0, 1.0)
    inflation = st.sidebar.slider("Inflation Rate (Annual %)", -2.0, 50.0, 3.0)

    life_exp_squared = life_exp ** 2

    # I construct the input dictionary for the model
    input_data = {
        'const': 1.0,
        'trade_openness': trade,
        'fdi_inflows': fdi,
        'capital_formation': capital,
        'life_expectancy': life_exp,
        'labor_force_part': labor,
        'population_growth': pop_growth,
        'inflation_rate': inflation,
        'life_expectancy_squared': life_exp_squared
    }

    input_df = pd.DataFrame([input_data])

    # I calculate the prediction and reverse the logarithmic transformation
    log_gdp_pred = model.predict(input_df).iloc[0]
    real_gdp_pred = np.exp(log_gdp_pred)

    # --- MAIN UI: THE METRIC ---
    st.subheader("Simulated Economic Output")
    st.metric(label="Predicted GDP per Capita (USD)", value=f"${real_gdp_pred:,.2f}")

    st.markdown("---")

    # --- VIZ 1: COEFFICIENT ANALYSIS ---
    st.subheader("1. Economic Drivers (Regression Coefficients)")
    st.markdown(
        "This chart displays the exact weight each variable has on the natural logarithm of GDP. It proves that population growth drags down wealth per capita, while labor participation drives it up.")

    # I extract the coefficients, dropping the mathematical constant
    coefs = model.params.drop('const')
    coef_df = pd.DataFrame({'Feature': coefs.index, 'Impact (Coefficient)': coefs.values})
    coef_df = coef_df.sort_values(by='Impact (Coefficient)')

    fig1 = px.bar(
        coef_df,
        x='Impact (Coefficient)',
        y='Feature',
        orientation='h',
        color='Impact (Coefficient)',
        color_continuous_scale=px.colors.diverging.RdBu
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- VIZ 2: POLYNOMIAL CURVE ---
    st.subheader("2. The Non-Linear Impact of Health")
    st.markdown(
        "By engineering a squared polynomial term, the model captures the diminishing economic returns of life expectancy. The red dot represents your current slider selection.")

    # generate a synthetic range of life expectancies to plot the curve
    le_range = np.linspace(50, 85, 100)
    curve_df = pd.DataFrame([input_data] * 100)
    curve_df['life_expectancy'] = le_range
    curve_df['life_expectancy_squared'] = le_range ** 2

    # predicting the GDP for this entire curve
    curve_preds = np.exp(model.predict(curve_df))

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(x=le_range, y=curve_preds, mode='lines', name='Predicted GDP Curve', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=[life_exp], y=[real_gdp_pred], mode='markers', name='Current Simulation',
                              marker=dict(color='red', size=12)))

    fig2.update_layout(xaxis_title="Life Expectancy (Years)", yaxis_title="Predicted GDP (USD)")
    st.plotly_chart(fig2, use_container_width=True)

    # --- VIZ 3: ACTUAL VS PREDICTED ---
    st.subheader("3. Model Accuracy (Actual vs. Predicted)")
    st.markdown(
        "I apply the model to the historical World Bank dataset. Dots closer to the dashed line represent countries where the model's prediction was highly accurate.")

    # I replicate the numeric coercion and imputation from the training pipeline
    features = list(input_data.keys())[1:]  # Exclude 'const'
    X_hist = df[features].copy()
    X_hist = X_hist.apply(pd.to_numeric, errors='coerce')
    X_hist = X_hist.fillna(X_hist.median()).fillna(0)
    X_hist.insert(0, 'const', 1.0)

    # I predict the GDP for every historical country
    df['Predicted_Log_GDP'] = model.predict(X_hist)
    df['Predicted_GDP'] = np.exp(df['Predicted_Log_GDP'])

    # I plot the real data against the model's predictions using a logarithmic scale

    fig3 = px.scatter(
        df,
        x='gdp_per_capita',
        y='Predicted_GDP',
        hover_name='country_name',
        log_x=True,
        log_y=True,
        opacity=0.7,
        labels={'gdp_per_capita': 'Actual GDP per Capita (Log Scale)',
                'Predicted_GDP': 'Predicted GDP per Capita (Log Scale)'}
    )

    # I add a 45-degree reference line to show perfect prediction
    min_val = min(df['gdp_per_capita'].min(), df['Predicted_GDP'].min())
    max_val = max(df['gdp_per_capita'].max(), df['Predicted_GDP'].max())
    fig3.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="Red", dash="dash"))

    st.plotly_chart(fig3, use_container_width=True)

else:
    st.error("I cannot find the required data or model files. I must run data.py, features.py, and model.py first.")