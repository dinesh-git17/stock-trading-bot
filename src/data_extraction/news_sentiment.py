import argparse
import io
import logging
import os
import signal
import threading
import traceback
from datetime import datetime

import feedparser
import pandas as pd
import praw  # Reddit API
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from yahoo_fin import news

from src.tools.utils import get_database_engine, handle_exceptions, setup_logging

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/news_sentiment.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)  # Ensure log directory exists
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)
console = Console()
logger.info("🚀 Logging setup complete.")

# ✅ API Keys & Configuration
API_KEYS = os.getenv("NEWS_API_KEY", "").replace(" ", "").split(",")
if not API_KEYS or API_KEYS == [""]:
    logger.error("❌ NEWS_API_KEY is missing or invalid.")
    console.print(
        Panel("❌ ERROR: NEWS_API_KEY is missing or invalid.", style="bold red")
    )
    exit(1)

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = "stock-bot"

DATA_DIR = "data/raw/news_sentiment/"
os.makedirs(DATA_DIR, exist_ok=True)
logger.info("📁 Data directory verified: %s", DATA_DIR)
console.print(Panel("📁 Data directory verified.", style="bold green"))

# ✅ Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()
logger.info("🔍 Sentiment Analyzer initialized.")
console.print(Panel("🔍 Sentiment Analyzer initialized.", style="bold cyan"))

# ✅ Multi-threading & API Settings
STOP_EXECUTION = threading.Event()
logger.info("⚡ Multi-threading setup complete.")
console.print(Panel("⚡ Multi-threading setup complete.", style="bold magenta"))


def log_error(exception, custom_message=""):
    """Logs and displays errors in a structured Rich Box with debug tips."""
    logger.error(f"❌ {custom_message} - {exception}", exc_info=True)
    error_message = (
        f"\n🚨 [ERROR] {custom_message}\n{'-'*50}\n{traceback.format_exc()}{'-'*50}\n"
    )
    console.print(Panel(error_message, title="🚨 ERROR DETECTED", style="bold red"))


def handle_exit(signal_received, frame):
    """Handles keyboard interrupt and stops all processing safely."""
    console.print(
        Panel("🛑 Process interrupted. Stopping all tasks...", style="bold red")
    )
    logger.warning("🛑 Process interrupted by user. Stopping all tasks...")
    STOP_EXECUTION.set()
    exit(0)


signal.signal(signal.SIGINT, handle_exit)


@handle_exceptions
def fetch_all_tickers():
    """Fetches all distinct tickers from the `stocks` table."""
    logger.info("🔄 Fetching all distinct tickers from database...")
    console.print(
        Panel("🔄 Fetching all distinct tickers from database...", style="bold blue")
    )
    engine = get_database_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT ticker FROM stocks")).fetchall()
        tickers = [row[0] for row in result]
        logger.info(f"✅ Retrieved {len(tickers)} tickers from database.")
        console.print(
            Panel(
                f"✅ Retrieved {len(tickers)} tickers from database.",
                style="bold green",
            )
        )
        return tickers
    except SQLAlchemyError as e:
        log_error(e, "Error fetching tickers from database.")
        return []


@handle_exceptions
def fetch_news_from_newsapi(ticker: str) -> pd.DataFrame:
    """Fetches news articles for a stock ticker from NewsAPI, switching API keys on 429 errors."""
    logger.info(f"🌍 Fetching NewsAPI data for {ticker}...")
    console.print(Panel(f"🌍 Fetching NewsAPI data for {ticker}...", style="bold blue"))

    api_keys = API_KEYS.copy()  # Create a mutable copy to remove exhausted keys
    while api_keys:
        api_key = api_keys.pop(
            0
        )  # Get the first available API key and remove it from the list
        params = {
            "q": ticker,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": api_key,
        }
        response = requests.get(
            "https://newsapi.org/v2/everything", params=params, timeout=10
        )

        if response.status_code == 200 and "articles" in response.json():
            articles = response.json().get("articles", [])
            df = pd.DataFrame(articles)

            # Ensure required columns exist before processing
            required_columns = ["publishedAt", "title", "description", "url", "source"]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = "Unknown"

            # Selecting only the required columns and handling missing data
            df = df.rename(
                columns={
                    "publishedAt": "published_at",
                    "title": "title",
                    "description": "description",
                    "url": "url",
                }
            )
            df["ticker"] = ticker

            # Ensure 'source' column exists before extracting name
            df["source_name"] = df["source"].apply(
                lambda x: (
                    x["name"] if isinstance(x, dict) and "name" in x else "Unknown"
                )
            )

            df["sentiment_score"] = df["title"].apply(
                lambda x: (
                    analyzer.polarity_scores(str(x))["compound"] if pd.notna(x) else 0
                )
            )
            df["created_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            df = df[
                [
                    "ticker",
                    "published_at",
                    "source_name",
                    "title",
                    "description",
                    "url",
                    "sentiment_score",
                    "created_at",
                ]
            ]

            logger.info(f"✅ NewsAPI: {len(df)} articles fetched for {ticker}.")
            console.print(
                Panel(f"✅ NewsAPI: {len(df)} articles fetched.", style="bold green")
            )
            return df
        elif response.status_code == 429:
            if api_keys:
                logger.warning(
                    f"⚠️ API key {api_key} hit rate limit (429). Removing and trying next key..."
                )
                console.print(
                    Panel(
                        f"⚠️ API key {api_key} hit rate limit. Trying next key...",
                        style="bold yellow",
                    )
                )
            continue

    log_error(
        Exception("All API keys hit rate limits"),
        f"Failed to fetch NewsAPI data for {ticker}. All keys exhausted.",
    )
    return pd.DataFrame(
        columns=[
            "ticker",
            "published_at",
            "source_name",
            "title",
            "description",
            "url",
            "sentiment_score",
            "created_at",
        ]
    )


@handle_exceptions
def fetch_news_from_yahoo(ticker: str) -> pd.DataFrame:
    """Fetches financial news for a stock ticker from Yahoo Finance and retains only relevant columns."""
    logger.info(f"🌍 Fetching Yahoo Finance news for {ticker}...")
    console.print(
        Panel(f"🌍 Fetching Yahoo Finance news for {ticker}...", style="bold blue")
    )
    try:
        articles = news.get_yf_rss(ticker)
        if not articles:
            logger.warning(f"⚠️ No Yahoo Finance articles found for {ticker}.")
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "published_at",
                    "source_name",
                    "title",
                    "description",
                    "url",
                    "sentiment_score",
                    "created_at",
                ]
            )

        df = pd.DataFrame(articles)

        # Ensure required columns exist before renaming
        if "published" not in df.columns:
            df["published"] = None  # Fill with None if missing
        if "title" not in df.columns:
            df["title"] = None
        if "summary" not in df.columns:
            df["summary"] = None
        if "link" not in df.columns:
            df["link"] = None

        # Selecting only the required columns and ensuring correct formatting
        df = df.rename(
            columns={
                "published": "published_at",
                "title": "title",
                "summary": "description",
                "link": "url",
            }
        )
        df["ticker"] = ticker
        df["source_name"] = "Yahoo Finance"
        df["sentiment_score"] = df["title"].apply(
            lambda x: analyzer.polarity_scores(str(x))["compound"] if pd.notna(x) else 0
        )
        df["created_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        df = df[
            [
                "ticker",
                "published_at",
                "source_name",
                "title",
                "description",
                "url",
                "sentiment_score",
                "created_at",
            ]
        ]

        logger.info(f"✅ Yahoo Finance: {len(df)} articles fetched for {ticker}.")
        console.print(
            Panel(f"✅ Yahoo Finance: {len(df)} articles fetched.", style="bold green")
        )
        return df

    except Exception as e:
        log_error(e, f"Failed to fetch Yahoo Finance news for {ticker}")
        return pd.DataFrame(
            columns=[
                "ticker",
                "published_at",
                "source_name",
                "title",
                "description",
                "url",
                "sentiment_score",
                "created_at",
            ]
        )


@handle_exceptions
def fetch_news_from_google_rss(ticker: str) -> pd.DataFrame:
    """Fetches stock news from Google News RSS feeds."""
    logger.info(f"🌍 Fetching Google News RSS for {ticker}...")
    console.print(
        Panel(f"🌍 Fetching Google News RSS for {ticker}...", style="bold blue")
    )
    url = f"https://news.google.com/rss/search?q={ticker}+stock"
    feed = feedparser.parse(url)
    articles = []

    for entry in feed.entries:
        title, description = entry.get("title", ""), entry.get("summary", "")
        sentiment_score = analyzer.polarity_scores(f"{title} {description}")["compound"]
        articles.append(
            {
                "ticker": ticker,
                "published_at": entry.get("published", ""),
                "source_name": "Google News",
                "title": title,
                "description": description,
                "url": entry.get("link", ""),
                "sentiment_score": sentiment_score,
                "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    console.print(
        Panel(
            f"✅ Google News RSS: {len(articles)} articles fetched.", style="bold green"
        )
    )
    return pd.DataFrame(articles)


@handle_exceptions
def fetch_news_from_reddit(ticker: str) -> pd.DataFrame:
    """Fetches discussions from Reddit finance forums and returns a DataFrame."""
    logger.info(f"🌍 Fetching Reddit discussions for {ticker}...")
    console.print(
        Panel(f"🌍 Fetching Reddit discussions for {ticker}...", style="bold blue")
    )
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )

        subreddit = reddit.subreddit("stocks")
        posts = subreddit.search(f"{ticker}", sort="new", limit=10)

        articles = []
        for post in posts:
            sentiment_score = analyzer.polarity_scores(post.title)["compound"]
            articles.append(
                {
                    "ticker": ticker,
                    "published_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "source_name": "Reddit",
                    "title": post.title,
                    "description": None,
                    "url": post.url,
                    "sentiment_score": sentiment_score,
                    "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )

        console.print(
            Panel(f"✅ Reddit: {len(articles)} posts fetched.", style="bold green")
        )
        return pd.DataFrame(articles)

    except Exception as e:
        log_error(e, f"Failed to fetch Reddit discussions for {ticker}")
        return pd.DataFrame()


@handle_exceptions
def save_dataframe_to_csv(df: pd.DataFrame, ticker: str):
    """Saves the given DataFrame to a CSV file in DATA_DIR with the format <ticker>_sentiment.csv."""
    try:
        if df.empty:
            logger.warning(
                "⚠️ Attempted to save an empty DataFrame for ticker: %s", ticker
            )
            console.print(
                Panel(
                    f"⚠️ Warning: The DataFrame for {ticker} is empty. Skipping save.",
                    style="bold yellow",
                )
            )
            return

        filename = f"{ticker}_sentiment.csv"
        file_path = os.path.join(DATA_DIR, filename)
        df.to_csv(file_path, index=False)
        logger.info("✅ Data saved successfully: %s", file_path)
        console.print(
            Panel(f"✅ Data saved successfully: {file_path}", style="bold green")
        )
    except Exception as e:
        log_error(e, f"Failed to save DataFrame to CSV for {ticker}")


@handle_exceptions
def convert_datetime_format(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the published_at column to a PostgreSQL-compatible format."""
    if "published_at" in df.columns:

        def parse_datetime(value):
            if pd.isna(value) or not isinstance(value, str):
                return None
            try:
                return datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %z").strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    try:
                        return datetime.strptime(
                            value, "%a, %d %b %Y %H:%M:%S GMT"
                        ).strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        return None

        df["published_at"] = df["published_at"].apply(parse_datetime)
    return df


@handle_exceptions
def insert_news_sentiment_data(df: pd.DataFrame):
    """Efficiently inserts cleaned news sentiment data into the PostgreSQL 'news_sentiment' table."""
    if df.empty:
        logger.warning("⚠️ No new news sentiment data to insert.")
        console.print(Panel("⚠️ Warning: No new data to insert.", style="bold yellow"))
        return

    engine = get_database_engine()
    try:
        with engine.connect() as conn:
            existing_urls = set(
                row[0]
                for row in conn.execute(
                    text("SELECT url FROM news_sentiment")
                ).fetchall()
            )

        # Filter out duplicates
        df = df[~df["url"].isin(existing_urls)]
        if df.empty:
            logger.info("✅ No new unique records to insert.")
            console.print(
                Panel("✅ No new unique records to insert.", style="bold green")
            )
            return

        # Ensure no missing values by replacing NaN with NULL strings
        df = df.fillna("NULL")

        # Properly escape text fields to avoid COPY command failures
        text_fields = ["title", "description", "source_name"]
        for field in text_fields:
            df[field] = (
                df[field]
                .astype(str)
                .apply(
                    lambda x: x.replace("\n", " ")
                    .replace("\t", " ")
                    .replace('"', "'")
                    .strip()
                )
            )

        # Convert DataFrame to CSV format for PostgreSQL COPY command with explicit escape character
        output = io.StringIO()
        df.to_csv(
            output, sep="\t", index=False, header=False, quoting=3, escapechar="\\"
        )  # quoting=3 (QUOTE_NONE), escapechar for special chars
        output.seek(0)

        conn = engine.raw_connection()
        try:
            cursor = conn.cursor()
            cursor.copy_from(
                output,
                "news_sentiment",
                sep="\t",
                columns=[
                    "ticker",
                    "published_at",
                    "source_name",
                    "title",
                    "description",
                    "url",
                    "sentiment_score",
                    "created_at",
                ],
            )
            conn.commit()
        finally:
            conn.close()  # ✅ Ensure the connection is properly closed

        logger.info(
            f"✅ Successfully inserted {len(df)} new records into news_sentiment."
        )
        console.print(
            Panel(
                f"✅ Successfully inserted {len(df)} new records.", style="bold green"
            )
        )
    except Exception as e:
        log_error(e, "Failed to insert news sentiment data into database.")


if __name__ == "__main__":
    logger.info("🚀 News Sentiment Analysis script started.")
    console.print(Panel("🚀 Fetching News Sentiment for Stocks...", style="bold green"))
    parser = argparse.ArgumentParser(
        description="Fetch & analyze news sentiment for stocks."
    )
    parser.add_argument(
        "--ticker", type=str, help="Run sentiment analysis for a specific ticker."
    )
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else fetch_all_tickers()
    if not tickers:
        console.print(
            Panel("❌ No tickers found in database. Exiting...", style="bold red")
        )
        logger.error("❌ No tickers found in database. Exiting...")
        exit(1)

    for ticker in tickers:
        if STOP_EXECUTION.is_set():
            logger.warning("🛑 Process interrupted. Exiting gracefully.")
            console.print(
                Panel("🛑 Process interrupted. Exiting gracefully.", style="bold red")
            )
            break

        df_newsapi = fetch_news_from_newsapi(ticker)
        df_google = fetch_news_from_google_rss(ticker)
        df_yahoo = fetch_news_from_yahoo(ticker)
        df_reddit = fetch_news_from_reddit(ticker)

        # Combine data from all sources
        all_data = pd.concat(
            [df_newsapi, df_google, df_yahoo, df_reddit], ignore_index=True
        )

        # Convert published_at to a consistent PostgreSQL-compatible format
        all_data = convert_datetime_format(all_data)

        # Save data to CSV
        save_dataframe_to_csv(all_data, ticker)

        # Insert cleaned data into the PostgreSQL 'news_sentiment' table
        insert_news_sentiment_data(all_data)
