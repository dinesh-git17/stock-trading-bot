import concurrent.futures
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime, timedelta

import psycopg2
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/historical_news_sentiment.log"
os.makedirs("data/logs", exist_ok=True)

with open(LOG_FILE, "a") as log_file:
    log_file.write("\n" * 5)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ Database Configuration
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# ✅ FMP API Configuration
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_NEWS_API_URL = "https://financialmodelingprep.com/api/v3/stock_news"

# ✅ Request Rate Limiting
REQUEST_DELAY = 2  # Base delay for API requests (in seconds)
MAX_RETRIES = 5  # Maximum retries before skipping a source
RANDOM_DELAY_RANGE = (1, 3)  # Random delay range to distribute API load


# ✅ Graceful Exit Handler
def handle_exit(signum, frame):
    console.print("\n[bold red]⚠ Process interrupted! Exiting ...[/bold red]")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


### **🚀 Smart API Request Function with Exponential Backoff**
def make_request(url, params=None):
    retries = 0
    delay = REQUEST_DELAY

    while retries < MAX_RETRIES:
        try:
            response = requests.get(url, params=params)
            if response.status_code == 429:
                logging.warning(f"⚠ API Rate Limit Hit. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                retries += 1
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ API Request Failed: {e}")
            retries += 1
            time.sleep(delay)
            delay *= 2
    return None  # Return None after max retries


### **🚀 Fetch Historical News from FMP**
def fetch_fmp_news(ticker, date):
    """
    Fetches historical news headlines from FMP for a given ticker and date.
    """
    params = {
        "apikey": FMP_API_KEY,
        "tickers": ticker,
        "limit": 10,  # Adjust based on API limits
    }
    data = make_request(FMP_NEWS_API_URL, params=params)

    if data:
        filtered_news = [
            {"title": article["title"], "publishedAt": article["publishedDate"]}
            for article in data
            if article["publishedDate"].startswith(date)
        ]
        return filtered_news

    return []


### **🚀 Sentiment Analysis**
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return round(analyzer.polarity_scores(text)["compound"], 3)


### **🚀 Check if Sentiment Data Exists (Avoid Duplicates)**
def sentiment_exists(cursor, ticker, date):
    """
    Checks if sentiment data for a given ticker and date already exists in the database.
    """
    cursor.execute(
        "SELECT EXISTS (SELECT 1 FROM news_sentiment WHERE ticker = %s AND published_at::date = %s)",
        (ticker, date),
    )
    return cursor.fetchone()[0]


### **🚀 Store Sentiment Data in Database**
def store_sentiment_data(cursor, ticker, date, articles):
    """
    Stores historical news sentiment data in the database.
    """
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
    if records:
        cursor.executemany(sql, records)
        return True
    return False


### **🚀 Process Historical Sentiment for a Single Ticker**
def process_ticker_sentiment(ticker, start_date, end_date, db_params):
    """
    Processes historical sentiment for a single ticker.
    """
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    total_days = (end_date - start_date).days + 1
    for single_date in (start_date + timedelta(n) for n in range(total_days)):
        date_str = single_date.date().isoformat()

        if sentiment_exists(cursor, ticker, date_str):
            continue

        articles = fetch_fmp_news(ticker, date_str)
        if articles and store_sentiment_data(cursor, ticker, date_str, articles):
            conn.commit()
            logging.info(f"✅ Stored news sentiment for {ticker} on {date_str}")

        time.sleep(random.uniform(*RANDOM_DELAY_RANGE))  # Add random delay

    cursor.close()
    conn.close()
    return f"✅ Completed {ticker}"


### **🚀 Process All Historical Sentiment in Parallel**
def process_historical_sentiment():
    db_params = {
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "host": DB_HOST,
        "port": DB_PORT,
    }

    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM stocks;")
    tickers = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    start_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
    end_date = datetime.today()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                process_ticker_sentiment, ticker, start_date, end_date, db_params
            ): ticker
            for ticker in tickers
        }
        for future in concurrent.futures.as_completed(futures):
            print(futures[future])


if __name__ == "__main__":
    process_historical_sentiment()
