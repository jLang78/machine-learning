# Spotify Streaming Forecaster

A machine learning system built to predict daily streaming volume for high-velocity music tracks using an XGBoost model. While I demonstrated this system on Ed Sheeran's "Shape of You," the underlying pipeline is designed as a scalable engine for the wider music industry.

## Value
Streaming numbers in the music industry can be viewed as volatile and unpredictable. The purspose of this project was to prove that they are highly modeled. 
I designed this system to serve as a decision support tool for three key use cases:

1.  **Baseline Benchmarking:** Accurately predicting "organic" performance augments methods to measure true ROI of marketing campaigns (actual streams - predicted baseline = marketing lift).
2.  **Inventory Planning:** Forecasts help distributors plan merchandise stock based on digital consumption trends.
3.  **Anomaly Detection:** The model acts as an early warning system. If actual streams deviate significantly from the forecast (e.g., a sudden 50% drop), it flags potential data feed issues or platform outages.

## Scalability: The "Franchise" Concept
"How is a model trained on an Ed Sheeran song useful for other artists?"

I built this project to serve as a bluprint. 
* **The Pipeline (src/):** This contains universal logic for feature engineering (lag extraction, seasonality, trend analysis) that applies to any artist.
* **The Model (.json):** This is the product.
* **Application:** To generate a forecast for a different artist (e.g., Taylor Swift), I simply need to feed the existing `src/train.py` pipeline a new raw data file. The system automatically learns the new artist's specific seasonality and decay curves without requiring code changes.

## Project Architecture
I structured the codebase for production deployment, separating logic from experimentation:

| Component | File | Description                                                                                                                                           |
| :--- | :--- |:------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Ingestion** | `src/features.py` | Automated feature engineering pipeline. Transforms raw CSVs into supervised learning datasets (creating Lags, Rolling Means, and Temporal Encodings). |
| **Blueprint** | `src/models.py` | Centralised model definition. Configures the XGBoost Regressor hyperparameters.                                                                       |
| **Training** | `src/train.py` | Orchestrates the split (Train/Test), trains the model, calculates error metrics, and serialises the model to disk.                                    |
| **Inference** | `src/predict.py` | The "live" system. Loads the saved model to predict streams for the next unknown day.                                                                 |
| **Reporting** | `src/evaluate.py` | An independent auditor that loads the model and generates visual performance reports.                                                                 |

## Results & Performance
I trained the model on historical data (2017–2020) and tested it against a holdout set (2021).

* **Mean Absolute Error (MAE):** +/- 21,910 streams
* **Daily Volume Baseline:** ~900,000 streams
* **Accuracy:** 97.6%
* **Key Discovery:** The model successfully learned the "weekend effect," autonomously identifying that streaming behavior shifts distinctly on Saturdays and Sundays without human intervention.

For visual validation, refer to the performance graph stored in `reports/figures/forecast_performance_pro.png`.

## How to Run
Prerequisites: pandas, xgboost, scikit-learn, matplotlib.

1.  **Train the System:**
    This ingests data, learns the patterns, and saves the model to the `models/` directory.
    ```bash
    python -m src.train
    ```

2.  **Generate a Performance Report:**
    Creates the "Actual vs. Forecast" graph in `reports/figures/`.
    ```bash
    python -m src.evaluate
    ```

3.  **Predict the Future:**
    Forecasts the stream count for the next day after the dataset ends.
    ```bash
    python -m src.predict
    ```

## Future Roadmap & Technical Implementation
While the current version is a robust baseline, I have identified two critical upgrades required to turn this into a more useful commercial system.

### 1. Predicting Long-Range Futures (e.g., 2026)
Currently, the model requires the immediate past (yesterday's streams) to predict the immediate future (tomorrow's streams). To predict a date in 2026 using a dataset that ends in 2021, I would implement one of two strategies:

* **Strategy A: Live Data Ingestion (preferred):** The model cannot skip five years of context. To predict 2026 accurately, the system must ingest the missing data (2022–2025). I would implement a scraper or API connector (e.g., Spotipy) in `src/features.py` to fetch the most recent 7 days of history before running inference.
* **Strategy B: Recursive Forecasting (fallback):** If recent data is unavailable, I would implement a recursive loop in `src/predict.py`. The model would predict Day 1, then use that prediction as the "actual" data to predict Day 2, and so on. *Note: This leads to error compounding and is not recommended for long horizons.*

### 2. Incorporating Exogenous Features
To improve accuracy during viral events or tours, I would expand the `preprocess_data` function in `src/features.py` to accept external signals.

* **Implementation:** I would merge the primary chart data with an external "Events" dataset.
* **New Features:**
    * `days_since_release`: A simple integer count to capture the natural decay curve of a song.
    * `is_on_tour`: A binary flag (1/0) derived from concert dates.
    * `marketing_spend`: A continuous variable representing daily ad spend.
    * `anything else`
* **Model Impact:** XGBoost would automatically learn the correlation between these flags and streaming spikes, allowing the model to anticipate non-seasonal uplifts.