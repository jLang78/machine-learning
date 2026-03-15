from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import sys
from pathlib import Path


# I explicitly add the project root to the system path so Python can find the 'src' module.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# I import the embedding model name from my centralised config file.
from src.config import EMBEDDING_MODEL_NAME


def train_topic_model(docs):
    # I initialise the lightweight embedding model optimised for my local hardware.
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # I use a CountVectorizer to strip out common English stopwords and capture 1- to 2-word phrases.
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

    # I initialise BERTopic with my custom components and set a low minimum topic size
    # to ensure it finds clusters even with smaller datasets.
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=5
    )

    # I train the model on the provided list of text documents.
    topics, probs = topic_model.fit_transform(docs)

    return topic_model, topics


def calculate_topic_drift(topic_model, docs, timestamps):
    # I calculate the dynamic topic representation across the provided timestamps.
    # This generates the dataframe required for the time-series visualisations.
    topics_over_time = topic_model.topics_over_time(docs, timestamps)

    return topics_over_time