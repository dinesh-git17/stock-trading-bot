import argparse
import logging
import os
import re
import signal
import threading
import time
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
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from yahoo_fin import news

from src.tools.utils import get_database_engine, setup_logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/logs/news_sentiment.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ✅ Load environment variables correctly
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/news_sentiment.log"
setup_logging(LOG_FILE)

console = Console()

# ✅ API Keys & Configuration
API_KEYS = os.getenv("NEWS_API_KEY", "").replace(" ", "").split(",")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = "stock-bot"

DATA_DIR = "data/raw/news_sentiment/"
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# ✅ Multi-threading & API Settings
STOP_EXECUTION = threading.Event()  # ✅ Global exit flag


def log_error(exception, custom_message=""):
    logger.info("Entering function: log_error")
    """Logs and displays errors in a structured Rich Box with debug tips."""
    error_message = (
        f"\n🚨 [ERROR] {custom_message}\n{'-'*50}\n{traceback.format_exc()}{'-'*50}\n"
    )
    debug_tips = """
🔍 **Debugging Tips**:
1️⃣ Check if the database is running.
2️⃣ Ensure no duplicate data is causing unique constraint errors.
3️⃣ Inspect the log file for details: data/logs/news_sentiment.log.
"""
    console.print(
        Panel(error_message + debug_tips, title="🚨 ERROR DETECTED", style="bold red")
    )
    logging.error(f"❌ {custom_message} - {exception}")


def handle_exit(signal_received, frame):
    logger.info("Entering function: handle_exit")
    """Handles keyboard interrupt and stops all processes safely."""
    console.print(
        Panel("🚨 Process interrupted by user. Stopping all tasks...", style="bold red")
    )
    logging.warning("🚨 Process interrupted by user. Stopping all tasks...")
    STOP_EXECUTION.set()  # ✅ Signal all threads to stop immediately
    exit(0)  # ✅ Force exit safely


def fetch_all_tickers():
    logger.info("Entering function: fetch_all_tickers")
    """Fetches all distinct tickers from the `stocks` table."""
    try:
        engine = get_database_engine()
        with engine.connect() as conn:
            query = text("SELECT DISTINCT ticker FROM stocks")
            result = conn.execute(query).fetchall()
        tickers = [row[0] for row in result]
        logging.info(f"✅ Retrieved {len(tickers)} tickers from database.")
        logger.info("Exiting function: fetch_all_tickers")
        return tickers
    except SQLAlchemyError as e:
        logger.error("An error occurred", exc_info=True)
        log_error(e, "Error fetching tickers from database.")
        logger.info("Exiting function: fetch_all_tickers")
        return []


def fetch_news_from_newsapi(ticker: str) -> pd.DataFrame:
    logger.info("Entering function: fetch_news_from_newsapi")
    """Fetches news articles for a stock ticker from NewsAPI and returns a DataFrame."""
    try:
        console.print(
            f"[bold blue]🌍 Fetching NewsAPI data for {ticker}...[/bold blue]"
        )
        params = {
            "q": ticker,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": API_KEYS[0],
        }
        response = requests.get(
            "https://newsapi.org/v2/everything", params=params, timeout=10
        )

        if response.status_code == 200:
            articles = response.json().get("articles", [])
            console.print(
                f"[green]✅ NewsAPI: {len(articles)} articles fetched.[/green]"
            )
            return pd.DataFrame(articles)

    except Exception as e:
        logger.error("An error occurred", exc_info=True)
        log_error(e, f"Failed to fetch data from NewsAPI for {ticker}")

    return pd.DataFrame()


def fetch_news_from_google_rss(ticker: str) -> pd.DataFrame:
    logger.info("Entering function: fetch_news_from_google_rss")
    """Fetches stock news from Google News RSS feeds and returns a properly formatted DataFrame."""
    try:
        console.print(
            f"[bold blue]🌍 Fetching Google News RSS for {ticker}...[/bold blue]"
        )
        url = f"https://news.google.com/rss/search?q={ticker}+stock"
        feed = feedparser.parse(url)

        articles = []
        for entry in feed.entries:
            title = entry.get("title", "")
            description = entry.get("summary", "")
            sentiment_score = analyzer.polarity_scores(
                str(title) + " " + str(description)
            )["compound"]

            articles.append(
                {
                    "ticker": ticker,
                    "published_at": entry.get(
                        "published", ""
                    ),  # ✅ Correct column name
                    "source_name": "Google News",
                    "title": title,
                    "description": description,
                    "url": entry.get("link", ""),
                    "sentiment_score": sentiment_score,
                    "created_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),  # ✅ Current timestamp
                }
            )

        console.print(
            f"[green]✅ Google News RSS: {len(articles)} articles fetched.[/green]"
        )
        return pd.DataFrame(articles)

    except Exception as e:
        logger.error("An error occurred", exc_info=True)
        log_error(e, f"Failed to fetch Google News RSS for {ticker}")

    return pd.DataFrame()


def fetch_news_from_yahoo(ticker: str) -> pd.DataFrame:
    logger.info("Entering function: fetch_news_from_yahoo")
    """Fetches financial news for a stock ticker from Yahoo Finance and returns a DataFrame."""
    try:
        console.print(
            f"[bold blue]🌍 Fetching Yahoo Finance news for {ticker}...[/bold blue]"
        )
        articles = news.get_yf_rss(ticker)
        console.print(
            f"[green]✅ Yahoo Finance: {len(articles)} articles fetched.[/green]"
        )
        logger.info("Exiting function: fetch_news_from_yahoo")
        return pd.DataFrame(articles)

    except Exception as e:
        logger.error("An error occurred", exc_info=True)
        log_error(e, f"Failed to fetch Yahoo Finance news for {ticker}")

    return pd.DataFrame()


# ✅ List of invalid words that should never be considered stock tickers
INVALID_TICKERS = {
    "ETF",
    "USD",
    "AI",
    "FTC",
    "AH",
    "US",
    "DOJ",
    "TSMC",
    "CEO",
    "IPO",
    "CPI",
    "I",
    "ITM",
}


def extract_ticker_from_title(title: str, default_ticker: str) -> str:
    """
    Extracts the stock ticker from a Reddit post title using regex.
    Ensures only valid stock tickers (1-5 uppercase letters) are extracted.
    Excludes common words that are not stock tickers.
    """
    # ✅ Use a stricter regex to match 2-5 uppercase letters (avoiding single letters)
    match = re.findall(r"\b[A-Z]{2,5}\b", title)

    # ✅ Remove words that are not actual stock tickers
    valid_tickers = [word for word in match if word not in INVALID_TICKERS]

    return (
        valid_tickers[0] if valid_tickers else default_ticker
    )  # Use first valid match or default ticker


def analyze_sentiment(text: str) -> float:
    """
    Analyzes sentiment of a given text using VADER Sentiment Analysis.
    Returns a sentiment score between -1 (negative) and 1 (positive).
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)["compound"]
    return sentiment_score


def fetch_news_from_reddit(ticker: str) -> pd.DataFrame:
    """
    Fetches discussions from Reddit finance forums and returns a DataFrame.
    Extracts the stock ticker from the post title if available.
    Ensures `published_at` is never NULL by assigning a default timestamp if missing.
    Calculates sentiment score for each post and sets `source_name = "Reddit"`.
    Ensures all required columns exist before inserting into the database.
    """
    logger.info("Entering function: fetch_news_from_reddit")
    try:
        console.print(
            f"[bold blue]🌍 Fetching Reddit discussions for {ticker}...[/bold blue]"
        )
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )

        subreddit = reddit.subreddit("stocks")
        posts = subreddit.search(f"{ticker}", sort="new", limit=10)

        articles = []
        missing_ticker_count = 0
        missing_date_count = 0

        for post in posts:
            extracted_ticker = extract_ticker_from_title(post.title, ticker)
            if not extracted_ticker:
                missing_ticker_count += 1

            # Ensure `published_at` is properly assigned
            if post.created_utc:
                published_at = datetime.utcfromtimestamp(post.created_utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            else:
                published_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                missing_date_count += 1

            # **Calculate Sentiment Score**
            sentiment_score = analyze_sentiment(post.title)

            articles.append(
                {
                    "ticker": extracted_ticker,
                    "published_at": published_at,  # Ensure NOT NULL
                    "source_name": "Reddit",  # Set source name
                    "title": post.title,
                    "description": None,  # Reddit posts don't have a description
                    "url": post.url,
                    "sentiment_score": sentiment_score,  # Store Sentiment Score
                    "created_at": datetime.utcnow().strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),  # Timestamp for data insertion
                }
            )

        if missing_ticker_count > 0:
            logger.warning(
                f"⚠️ {missing_ticker_count} articles missing a ticker. Default `{ticker}` used."
            )

        if missing_date_count > 0:
            logger.warning(
                f"⚠️ {missing_date_count} articles missing `published_at`. Default timestamp assigned."
            )

        # Convert to DataFrame
        df = pd.DataFrame(articles)

        # **Ensure No NULL Published Dates (Fixed Chained Assignment)**
        df["published_at"] = df["published_at"].fillna(
            datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        # **Ensure all required columns exist before inserting into database**
        required_columns = [
            "ticker",
            "published_at",
            "source_name",
            "title",
            "description",
            "url",
            "sentiment_score",
            "created_at",
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = None  # Add missing columns with None values

        console.print(f"[green]✅ Reddit: {len(df)} posts fetched.[/green]")

        return df

    except Exception as e:
        logger.error("An error occurred", exc_info=True)
        log_error(e, f"Failed to fetch Reddit discussions for {ticker}")

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


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)

    parser = argparse.ArgumentParser(
        description="Fetch & analyze news sentiment for stocks."
    )
    parser.add_argument(
        "--ticker", type=str, help="Run sentiment analysis for a specific ticker."
    )
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else fetch_all_tickers()

    if not tickers:
        console.print("[red]❌ No tickers found in database. Exiting...[/red]")
        exit(1)

    console.print("[bold green]🚀 Fetching News Sentiment for Stocks...[/bold green]")

    try:
        # ✅ Fixing list comprehension issue and ensuring valid DataFrames
        all_dfs = [
            df
            for ticker in tickers
            for df in [
                fetch_news_from_newsapi(ticker),
                fetch_news_from_google_rss(ticker),
                fetch_news_from_yahoo(ticker),
                fetch_news_from_reddit(ticker),
            ]
            if not df.empty
        ]
        all_data = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        # ✅ Filtering only required columns before inserting into the database

        if not all_data.empty:
            engine = get_database_engine()

            # ✅ Select only the required columns
            valid_columns = [
                "ticker",
                "published_at",
                "source_name",
                "title",
                "description",
                "url",
                "sentiment_score",
                "created_at",
            ]

            all_data = all_data[
                valid_columns
            ]  # ✅ Ensure only these columns are inserted

            # ✅ Step 2: Remove any rows where `ticker` is NULL
            missing_ticker_count = all_data["ticker"].isna().sum()

            if missing_ticker_count > 0:
                console.print(
                    f"[yellow]⚠️ Warning: {missing_ticker_count} articles are missing a ticker and will be skipped.[/yellow]"
                )
                all_data = all_data.dropna(subset=["ticker"])

            # ✅ Step 3: Fetch existing URLs to avoid duplicate insertions
            try:
                with engine.connect() as conn:
                    existing_urls = set(
                        row[0]
                        for row in conn.execute(
                            text("SELECT url FROM news_sentiment")
                        ).fetchall()
                    )
            except Exception as e:
                logger.error("An error occurred", exc_info=True)
                log_error(e, "Error fetching existing URLs from database.")
                existing_urls = set()

            # ✅ Step 4: Remove duplicate URLs before inserting
            all_data = all_data[~all_data["url"].isin(existing_urls)]

            # ✅ Step 5: Insert into the database only if there are valid rows
            if not all_data.empty:
                required_columns = [
                    "ticker",
                    "published_at",
                    "source_name",
                    "title",
                    "description",
                    "url",
                    "sentiment_score",
                    "created_at",
                ]
                for col in required_columns:
                    if col not in all_data.columns:
                        all_data[col] = (
                            None  # Ensure missing columns are created with None values
                        )
                        all_data.to_sql(
                            "news_sentiment", engine, if_exists="append", index=False
                        )
                        console.print(
                            Panel(
                                "[green]✅ Sentiment Analysis Complete! Data saved in DB.[/green]",
                                style="bold green",
                            )
                        )
            else:
                console.print(
                    "[yellow]⚠️ No new valid data to insert. Skipping database update.[/yellow]"
                )

    except Exception as e:
        logger.error("An error occurred", exc_info=True)
        log_error(e, "Unexpected error in the main execution.")
