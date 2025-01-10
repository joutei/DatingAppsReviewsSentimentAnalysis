import os
import datetime
import time
import requests
import pandas as pd
from transformers import pipeline
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import requests_cache
from retry_requests import retry
import hopsworks
import hsfs
from pathlib import Path
from google_play_scraper import reviews

def get_periodic_reviews(app_id, lang, country, start_date, end_date):
    """
    Fetches periodic reviews from the dating app.

    Parameters:
    - app_id: str, the app's package name.
    - lang: str, the language code (e.g., 'en').
    - country: str, the country code (e.g., 'us').
    - start_date: str, the start date for fetching reviews (YYYY-MM-DD).
    - end_date: str, the end date for fetching reviews (YYYY-MM-DD).
    
    Returns:
    - Pandas DataFrame containing the weekly reviews.
    """
    try:
        # Fetch reviews
        result, _ = reviews(
            app_id=app_id,
            lang=lang,
            country=country,
            count=1000  # Fetch a reasonable number of reviews
        )

        # Convert fetched reviews into a DataFrame
        df = pd.DataFrame(result)

        # Filter reviews by date range
        df['at'] = pd.to_datetime(df['at'])
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_reviews = df[(df['at'] >= start_date) & (df['at'] <= end_date)]

        # Keep only relevant keys
        relevant_keys = ['reviewId', 'content', 'score', 'at', 'thumbsUpCount']
        filtered_reviews = filtered_reviews[relevant_keys]

        print(f"Fetched {len(filtered_reviews)} reviews from {start_date} to {end_date}.")
        return filtered_reviews

    except Exception as e:
        print(f"Error while fetching reviews: {e}")
        return pd.DataFrame()

def get_last_week_reviews(app_id, lang, country, count=5000):
    """
    Fetches reviews from the last 7 days for a given app.

    Parameters:
    - app_id: str, the app's package name.
    - lang: str, the language code (e.g., 'en').
    - country: str, the country code (e.g., 'us').
    - count: int, number of reviews to fetch (default is 1000).

    Returns:
    - Pandas DataFrame containing reviews from the last 7 days.
    """
    try:
        # Calculate the date range for the last 7 days
        end_date = datetime.now()  # Current date
        start_date = end_date - timedelta(days=7)  # 7 days before today

        # Format dates as strings
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')

        # Fetch reviews
        result, _ = reviews(
            app_id=app_id,
            lang=lang,
            country=country,
            count=count  # Fetch the specified number of reviews
        )

        if not result:  # Handle empty responses
            print("No reviews found.")
            return pd.DataFrame()

        # Convert fetched reviews into a DataFrame
        df = pd.DataFrame(result)

        # Filter reviews by the last 7 days
        df['at'] = pd.to_datetime(df['at'])
        filtered_reviews = df[(df['at'] >= pd.to_datetime(start_date_str)) & (df['at'] <= pd.to_datetime(end_date_str))]

        # Keep only relevant keys
        relevant_keys = ['reviewId', 'content', 'score', 'at', 'thumbsUpCount']
        filtered_reviews = filtered_reviews[relevant_keys]

        print(f"Fetched {len(filtered_reviews)} reviews from {start_date_str} to {end_date_str}.")
        return filtered_reviews

    except Exception as e:
        print(f"Error while fetching reviews: {e}")
        return pd.DataFrame()

def get_last_30_days_reviews(app_id, lang, country):
    """
    Fetches reviews from the last 30 days for a given app.

    Parameters:
    - app_id: str, the app's package name.
    - lang: str, the language code (e.g., 'en').
    - country: str, the country code (e.g., 'us').

    Returns:
    - Pandas DataFrame containing reviews from the last 30 days.
    """
    try:
        # Calculate the date range for the last 30 days
        end_date = datetime.now()  # Current date
        start_date = end_date - timedelta(days=30)  # 30 days before today

        # Format dates as strings
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')

        # Fetch reviews
        result, _ = reviews(
            app_id=app_id,
            lang=lang,
            country=country,
            count=1000  # Fetch a reasonable number of reviews
        )

        # Convert fetched reviews into a DataFrame
        df = pd.DataFrame(result)

        # Filter reviews by the last 30 days
        df['at'] = pd.to_datetime(df['at'])
        filtered_reviews = df[(df['at'] >= pd.to_datetime(start_date_str)) & (df['at'] <= pd.to_datetime(end_date_str))]

        # Keep only relevant keys
        relevant_keys = ['reviewId', 'content', 'score', 'at', 'thumbsUpCount']
        filtered_reviews = filtered_reviews[relevant_keys]

        print(f"Fetched {len(filtered_reviews)} reviews from {start_date_str} to {end_date_str}.")
        return filtered_reviews

    except Exception as e:
        print(f"Error while fetching reviews: {e}")
        return pd.DataFrame()

def trigger_request(url:str):
    response = requests.get(url)
    if response.status_code == 200:
        # Extract the JSON content from the response
        data = response.json()
    else:
        print("Failed to retrieve data. Status Code:", response.status_code)
        raise requests.exceptions.RequestException(response.status_code)

    return data

def classify_topics_with_sentiment(df: pd.DataFrame, candidate_topics: list, model_name: str = "facebook/bart-large-mnli"):
    """
    Classifies reviews into predefined topics and groups them by sentiment.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'content' and 'sentiment'.
    - candidate_topics (list): List of candidate topics for zero-shot classification.
    - model_name (str): Hugging Face model name for zero-shot classification.

    Returns:
    - df (pd.DataFrame): DataFrame with an additional 'topic' column.
    - topic_counts (pd.DataFrame): Count of topics by sentiment.
    """
    # Initialize the zero-shot classifier
    classifier = pipeline("zero-shot-classification", model=model_name)

    # Define a function to classify topics
    def classify_topic(text):
        result = classifier(text, candidate_labels=candidate_topics)
        return result['labels'][0]  # Return the top predicted topic

    # Apply the topic classification
    df['topic'] = df['content'].apply(classify_topic)

    # Group by sentiment and topic to get counts
    topic_counts = df.groupby(['sentiment', 'topic']).size().unstack(fill_value=0)

    return df, topic_counts


def plot_sentiment_topics(topic_counts: pd.DataFrame, file_path: str):
    """
    Plots the topics grouped by sentiment as a stacked bar plot.

    Parameters:
    - topic_counts (pd.DataFrame): Count of topics by sentiment.
    - file_path (str): Path to save the plot.

    Returns:
    - plt: The Matplotlib plot object.
    """
    # Plot the stacked bar chart
    ax = topic_counts.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis')
    ax.set_title("Topics by Sentiment", fontsize=16)
    ax.set_xlabel("Sentiment", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Save the plot
    plt.tight_layout()
    plt.savefig(file_path)

    return plt



def delete_feature_groups(fs, name):
    try:
        for fg in fs.get_feature_groups(name):
            fg.delete()
            print(f"Deleted {fg.name}/{fg.version}")
    except hsfs.client.exceptions.RestAPIError:
        print(f"No {name} feature group found")

def delete_feature_views(fs, name):
    try:
        for fv in fs.get_feature_views(name):
            fv.delete()
            print(f"Deleted {fv.name}/{fv.version}")
    except hsfs.client.exceptions.RestAPIError:
        print(f"No {name} feature view found")

def delete_models(mr, name):
    models = mr.get_models(name)
    if not models:
        print(f"No {name} model found")
    for model in models:
        model.delete()
        print(f"Deleted model {model.name}/{model.version}")

def delete_secrets(proj, name):
    secrets = secrets_api(proj.name)
    try:
        secret = secrets.get_secret(name)
        secret.delete()
        print(f"Deleted secret {name}")
    except hopsworks.client.exceptions.RestAPIError:
        print(f"No {name} secret found")

# WARNING - this will wipe out all your feature data and models
def purge_project(proj):
    fs = proj.get_feature_store()
    mr = proj.get_model_registry()

    # Delete Feature Views before deleting the feature groups
    delete_feature_views(fs, "air_quality_fv")

    # Delete ALL Feature Groups
    delete_feature_groups(fs, "air_quality")
    delete_feature_groups(fs, "weather")
    delete_feature_groups(fs, "aq_predictions")

    # Delete all Models
    delete_models(mr, "air_quality_xgboost_model")
    delete_secrets(proj, "SENSOR_LOCATION_JSON")


def secrets_api(proj):
    host = "c.app.hopsworks.ai"
    api_key = os.environ.get('HOPSWORKS_API_KEY')
    conn = hopsworks.connection(host=host, project=proj, api_key_value=api_key)
    return conn.get_secrets_api()


def check_file_path(file_path):
    my_file = Path(file_path)
    if my_file.is_file() == False:
        print(f"Error. File not found at the path: {file_path} ")
    else:
        print(f"File successfully found at the path: {file_path}")

