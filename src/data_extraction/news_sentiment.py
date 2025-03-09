import os
import requests
import psycopg2
import logging
import time
import sys
from dotenv import load_dotenv
from textblob import TextBlob
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/news_sentiment.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# Database Configuration
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# API Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"


def fetch_stock_news(ticker, max_retries=3):
    """Fetches recent news articles for a stock ticker, handling rate limits."""
    params = {
        "q": ticker,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,  # Get latest 5 articles
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(NEWS_API_URL, params=params)

            # Handle rate limiting (429 Too Many Requests)
            if response.status_code == 429:
                wait_time = int(
                    response.headers.get("Retry-After", 5)
                )  # Default wait = 5s
                logging.warning(
                    f"Rate limit exceeded for {ticker}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)  # Wait before retrying
                retries += 1
                continue  # Retry request

            response.raise_for_status()
            return response.json().get("articles", [])

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch news for {ticker}: {e}")
            return []

    logging.error(f"Max retries reached for {ticker}. Skipping...")
    return []  # Return empty if retries fail


def analyze_sentiment(text):
    """Analyzes sentiment of a news article using TextBlob."""
    sentiment_score = TextBlob(
        text
    ).sentiment.polarity  # Ranges from -1 (negative) to 1 (positive)
    return round(sentiment_score, 3)


def store_sentiment_data(cursor, ticker, articles):
    """Stores news sentiment data in the database."""
    sql = """
        INSERT INTO news_sentiment (ticker, published_at, title, sentiment_score)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (ticker, published_at) DO NOTHING;
    """

    records = [
        (
            ticker,
            article["publishedAt"],
            article["title"],
            analyze_sentiment(article["title"]),
        )
        for article in articles
    ]

    cursor.executemany(sql, records)


def process_news_sentiment():
    """Fetches news articles, analyzes sentiment, and stores results."""
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT ticker FROM stocks;")
    tickers = [row[0] for row in cursor.fetchall()]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Fetching & analyzing news sentiment...[/bold yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("sentiment", total=len(tickers))

        for ticker in tickers:
            articles = fetch_stock_news(ticker)
            if articles:
                store_sentiment_data(cursor, ticker, articles)
                conn.commit()
                logging.info(f"Stored news sentiment for {ticker}")

            time.sleep(1)  # ✅ Add delay between API requests to avoid rate limiting
            progress.update(task, advance=1)

            # Check if the last fetch resulted in a rate limit
            if (
                not articles
                and "Rate limit exceeded" in open("data/logs/news_sentiment.log").read()
            ):
                console.print(
                    "[bold red]❌ API Rate Limit exceeded. Storing available data and exiting...[/bold red]"
                )
                logging.error(
                    "API Rate Limit exceeded. Exiting safely while storing available data."
                )
                break  # Stop processing further tickers

    conn.commit()
    cursor.close()
    conn.close()
    console.print(
        "[bold green]✅ News sentiment analysis completed with available data![/bold green]"
    )
    sys.exit(0)  # Exit safely


if __name__ == "__main__":
    process_news_sentiment()
