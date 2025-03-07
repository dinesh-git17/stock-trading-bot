import os
import requests
import psycopg2
import logging
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

# API Configuration (Replace with actual API Key)
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"


def connect_db():
    """Connects to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        return conn
    except Exception as e:
        console.print(f"[bold red]❌ Database connection error:[/bold red] {e}")
        logging.error(f"Database connection error: {e}")
        return None


def fetch_stock_news(ticker):
    """Fetches recent news articles related to a stock ticker."""
    params = {
        "q": ticker,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,  # Get the latest 5 articles
    }

    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()
        return response.json().get("articles", [])

    except Exception as e:
        logging.error(f"Failed to fetch news for {ticker}: {e}")
        return []


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
    conn = connect_db()
    if not conn:
        return
    cursor = conn.cursor()

    # Fetch stock tickers from the database
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

            progress.update(task, advance=1)

    cursor.close()
    conn.close()
    console.print("[bold green]✅ News sentiment analysis completed![/bold green]")


if __name__ == "__main__":
    process_news_sentiment()
