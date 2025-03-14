import argparse
import logging
import os
import signal
import threading
import time
import traceback

import pandas as pd
import requests
from dotenv import load_dotenv
from rich.console import Console
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.tools.utils import get_database_engine, setup_logging

# ✅ Load environment variables correctly
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/news_sentiment.log"
setup_logging(LOG_FILE)

console = Console()

# ✅ Fix API Key Loading & Splitting
API_KEYS = os.getenv("NEWS_API_KEY", "").replace(" ", "").split(",")

# ✅ Check if API Keys Are Loaded
if not API_KEYS or API_KEYS == [""]:
    console.print("[error]🚨 No API keys found! Check .env file.[/error]")
    logging.error("🚨 No API keys found! Check .env file.")
    exit(1)  # ✅ Stop execution if no API keys found

console.print(f"[info]🔑 Loaded API Keys: {', '.join(API_KEYS)}")
logging.info(f"🔑 Loaded API Keys: {', '.join(API_KEYS)}")

DATA_DIR = "data/raw/news_sentiment/"
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ API Configuration
API_ENDPOINT = "https://newsapi.org/v2/everything"

# ✅ Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# ✅ Multi-threading & API Settings
THREADS = 5
MAX_RETRIES = 3
RATE_LIMIT_SLEEP = 60  # ✅ Sleep time when hitting API limit
STOP_EXECUTION = threading.Event()  # ✅ Global exit flag


def log_error(exception):
    """Logs and displays errors in a structured format."""
    error_message = f"\n🚨 [ERROR] An exception occurred:\n{'-'*50}\n{traceback.format_exc()}{'-'*50}\n"
    console.print(f"[error]{error_message}[/error]")
    logging.error(error_message)


def handle_exit(signal_received, frame):
    """Handles keyboard interrupt and stops all processes safely."""
    console.print(
        "\n[error]🚨 Process interrupted by user. Stopping all tasks...[/error]"
    )
    logging.error("🚨 Process interrupted by user. Stopping all tasks...")
    STOP_EXECUTION.set()  # ✅ Signal all threads to stop immediately
    exit(0)  # ✅ Force exit safely


def analyze_sentiment(text: str) -> float:
    """Analyzes sentiment of the given text using VADER sentiment analysis.

    Args:
        text (str): The text (news article title) to analyze.

    Returns:
        float: Sentiment score between -1 (negative) and 1 (positive).
    """
    try:
        if not text:
            return 0.0
        sentiment = analyzer.polarity_scores(text)
        return sentiment["compound"]  # ✅ Extract compound sentiment score
    except Exception as e:
        log_error(e)
        return 0.0


def fetch_news(ticker: str) -> list:
    """Fetches news articles for a stock ticker with API failover."""
    try:
        params = {
            "q": ticker,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
        }

        for attempt in range(MAX_RETRIES):
            for api_key in list(
                API_KEYS
            ):  # ✅ Iterate over a copy to allow modification
                if STOP_EXECUTION.is_set():  # ✅ Stop if exit flag is triggered
                    return []

                console.print(f"[info]🌍 {ticker}: Using API Key {api_key[:5]}*****")
                logging.info(f"🌍 {ticker}: Using API Key {api_key[:5]}*****")

                try:
                    params["apiKey"] = api_key
                    response = requests.get(API_ENDPOINT, params=params, timeout=10)

                    if response.status_code == 200:
                        articles = response.json().get("articles", [])
                        logging.info(
                            f"✅ {ticker}: {len(articles)} articles fetched using API Key: {api_key[:5]}..."
                        )
                        console.print(
                            f"[success]✅ {ticker}: {len(articles)} articles fetched[/success]"
                        )
                        return articles

                except Exception:
                    log_error(Exception)

    except Exception:
        log_error(Exception)

    return []


def get_tickers_from_db() -> list:
    """Retrieves stock tickers from the database."""
    try:
        engine = get_database_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT ticker FROM stocks")).fetchall()
        return [row[0] for row in result]
    except SQLAlchemyError:
        log_error(SQLAlchemyError)
        return []


def save_to_database(df: pd.DataFrame):
    """Saves processed sentiment data into the database."""
    try:
        engine = get_database_engine()
        with engine.connect() as conn:
            df.to_sql("news_sentiment", conn, if_exists="append", index=False)
        logging.info("✅ Sentiment data stored in database.")
    except SQLAlchemyError:
        log_error(SQLAlchemyError)


def process_ticker_news(ticker: str) -> pd.DataFrame:
    """Fetches and analyzes news sentiment for a given ticker."""
    try:
        articles = fetch_news(ticker)
        if not articles:
            return pd.DataFrame()

        data = []
        for article in articles:
            sentiment_score = analyze_sentiment(
                article.get("title", "")
            )  # ✅ Now correctly processes sentiment
            data.append(
                {
                    "ticker": ticker,
                    "published_at": article.get("publishedAt"),
                    "source_name": (
                        article["source"]["name"]
                        if isinstance(article.get("source"), dict)
                        else None
                    ),  # ✅ Extract only "name"
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "url": article.get("url"),
                    "sentiment_score": sentiment_score,
                }
            )

        df = pd.DataFrame(data)
        df["published_at"] = pd.to_datetime(
            df["published_at"]
        )  # ✅ Ensure proper timestamp format
        return df
    except Exception as e:
        log_error(e)
        return pd.DataFrame()


def run_news_sentiment_analysis(tickers: list):
    """Runs the news sentiment analysis pipeline."""
    console.print("[bold blue]📡 Fetching & Analyzing News Sentiment...[/bold blue]")

    try:
        article_data = []
        for ticker in tickers:
            articles = process_ticker_news(ticker)
            if not articles.empty:  # ✅ Ensure it's a valid list
                article_data.append(articles)

        if not article_data:
            console.print(
                "[warning]⚠️ No news sentiment data available for any tickers.[/warning]"
            )
            logging.warning("⚠️ No news sentiment data available.")
            return

        all_data = pd.concat(
            article_data, ignore_index=True
        )  # ✅ Concatenate valid data
        save_to_database(all_data)
        logging.info("✅ Sentiment analysis complete and data stored.")
        console.print(
            "[green]✅ Sentiment Analysis Complete! Data saved in DB.[/green]"
        )

    except KeyboardInterrupt:
        handle_exit(None, None)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)  # ✅ Register Safe Exit on `Ctrl+C`

    parser = argparse.ArgumentParser(
        description="Fetch & analyze news sentiment for stocks."
    )
    parser.add_argument(
        "--ticker", type=str, help="Run sentiment analysis for a specific ticker."
    )
    args = parser.parse_args()

    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = get_tickers_from_db()

    run_news_sentiment_analysis(tickers)
