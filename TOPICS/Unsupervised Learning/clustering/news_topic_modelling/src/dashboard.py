import sys
from pathlib import Path

# I explicitly add the project root to the system path so Python can find the 'src' module.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
import glob
import json

from src.config import RAW_DATA_DIR
from src.features import process_raw_to_clean
from src.model import train_topic_model, calculate_topic_drift
from src.visualise import generate_topic_barchart, generate_distance_map, generate_drift_chart

st.set_page_config(page_title="News Pulse Tracker", layout="wide")

st.title("News Pulse: Unsupervised Topic Modeling")
st.write("This dashboard analyses technology news to discover hidden topics and track their drift over time.")


@st.cache_data
def load_and_clean_data():
    # I find all JSON files and stitch them together.
    file_pattern = str(RAW_DATA_DIR / "*_technology_news.json")
    all_files = glob.glob(file_pattern)

    if not all_files:
        return pd.DataFrame()

    all_articles = []
    for file_path in all_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_articles.extend(data.get('articles', []))

    df_raw = pd.DataFrame(all_articles)
    df_clean = process_raw_to_clean(df_raw)
    df_clean['date'] = pd.to_datetime(df_clean['publishedAt']).dt.date

    return df_clean


@st.cache_resource
def run_topic_modeling(docs, timestamps):
    # I train the model and calculate the time-series drift.
    topic_model, topics = train_topic_model(docs)
    topics_over_time = calculate_topic_drift(topic_model, docs, timestamps)
    return topic_model, topics_over_time


df = load_and_clean_data()

if df.empty:
    st.warning("No data found. Please run src/data.py to collect news articles.")
else:
    docs = df['text_cleaned'].tolist()
    timestamps = df['date'].tolist()

    with st.spinner("Training model and calculating topic drift. This may take a minute..."):
        model, drift_df = run_topic_modeling(docs, timestamps)

    # --- THE NEW SIDEBAR UI -
    # I create a sidebar for user controls.
    st.sidebar.header("Dashboard Controls")

    # I retrieve the list of topics the model discovered.
    topic_info = model.get_topic_info()

    # filter out Topic -1 because it represents unclustered outlier noise.
    valid_topics = topic_info[topic_info['Topic'] != -1]

    # creating a dictionary mapping the Topic ID to its generated Name for the dropdown UI.
    topic_mapping = dict(zip(valid_topics['Topic'], valid_topics['Name']))

    # I render the multiselect widget in the sidebar.
    selected_topic_names = st.sidebar.multiselect(
        "Select specific topics to isolate on the Drift Chart:",
        options=list(topic_mapping.values()),
        default=list(topic_mapping.values())[:3]  # I default to showing the top 3 topics
    )

    #converting the user's selected string names back into the integer IDs that BERTopic requires.
    selected_topic_ids = [k for k, v in topic_mapping.items() if v in selected_topic_names]

    st.success(f"Model trained successfully on {len(df)} articles across {df['date'].nunique()} unique days.")

    tab1, tab2, tab3 = st.tabs(["Topic Drift (Time Series)", "Top Topics (Bar Chart)", "Topic Clusters (Map)"])

    with tab1:
        st.subheader("How Topics Evolve Over Time")
        # passing the user's selected IDs into my visualisation function.
        fig_drift = generate_drift_chart(model, drift_df, selected_topics=selected_topic_ids)
        st.plotly_chart(fig_drift, use_container_width=True)

    with tab2:
        st.subheader("Most Important Keywords per Topic")
        fig_bar = generate_topic_barchart(model)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("Semantic Distance Between Topics")
        fig_map = generate_distance_map(model)
        st.plotly_chart(fig_map, use_container_width=True)