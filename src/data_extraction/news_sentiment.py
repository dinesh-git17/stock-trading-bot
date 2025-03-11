import json
import logging
import os
import sys
import time

import psycopg2
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from textblob import TextBlob

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/news_sentiment.log"
os.makedirs("data/logs", exist_ok=True)

# ✅ Insert 5 blank lines before logging new logs
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

# ✅ API Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
STOCKTWITS_API_URL = "https://api.stocktwits.com/api/2/streams/symbol/"
REDDIT_API_URL = "https://www.reddit.com/r/stocks/search.json"
TWITTER_API_URL = "https://api.twitter.com/2/tweets/search/recent"

NEWS_API_URL = "https://newsapi.org/v2/everything"
FINNHUB_API_URL = "https://finnhub.io/api/v1/news"
YAHOO_NEWS_API_URL = "https://yahoo-finance15.p.rapidapi.com/api/yahoo/ne/news"

# ✅ Backup Sources List (Fallback Order)
NEWS_SOURCES = [
    "newsapi",  # Primary
    "finnhub",  # Backup 1
    "yahoo_finance",  # Backup 2
    "stocktwits",  # Backup 3
    "reddit",  # Backup 4
    "twitter",  # Backup 5
]


### **🚀 Fetch Stock News with Backup Sources**
def fetch_stock_news(ticker, max_retries=3):
    """
    Fetches recent news articles for a stock ticker using multiple sources.
    Falls back to backup sources if the primary one fails.
    """
    for source in NEWS_SOURCES:
        retries = 0
        while retries < max_retries:
            try:
                if source == "newsapi":
                    params = {
                        "q": ticker,
                        "apiKey": NEWS_API_KEY,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 5,
                    }
                    response = requests.get(NEWS_API_URL, params=params)

                elif source == "finnhub":
                    params = {"category": "general", "token": FINNHUB_API_KEY}
                    response = requests.get(FINNHUB_API_URL, params=params)

                elif source == "yahoo_finance":
                    headers = {
                        "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com",
                        "x-rapidapi-key": os.getenv("YAHOO_FINANCE_API_KEY"),
                    }
                    response = requests.get(YAHOO_NEWS_API_URL, headers=headers)

                elif source == "stocktwits":
                    response = requests.get(f"{STOCKTWITS_API_URL}{ticker}.json")

                elif source == "reddit":
                    params = {"q": ticker, "sort": "new", "limit": 5}
                    headers = {"User-Agent": "stock-sentiment-bot"}
                    response = requests.get(
                        REDDIT_API_URL, params=params, headers=headers
                    )

                elif source == "twitter":
                    headers = {
                        "Authorization": f"Bearer {os.getenv('TWITTER_BEARER_TOKEN')}",
                    }
                    params = {"query": f"${ticker}", "max_results": 5}
                    response = requests.get(
                        TWITTER_API_URL, params=params, headers=headers
                    )

                # ✅ Handle rate limiting (429 Too Many Requests)
                if response.status_code == 429:
                    wait_time = int(response.headers.get("Retry-After", 5))
                    logging.warning(
                        f"⚠ Rate limit exceeded for {ticker} using {source}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    retries += 1
                    continue  # Retry request

                response.raise_for_status()

                # ✅ Extract articles based on source
                if source == "newsapi":
                    return response.json().get("articles", [])
                elif source == "finnhub":
                    return [
                        {"title": news["headline"], "publishedAt": news["datetime"]}
                        for news in response.json()
                    ]
                elif source == "yahoo_finance":
                    return [
                        {
                            "title": news["title"],
                            "publishedAt": news["providerPublishTime"],
                        }
                        for news in response.json()
                    ]
                elif source == "stocktwits":
                    return [
                        {"title": msg["body"], "publishedAt": msg["created_at"]}
                        for msg in response.json().get("messages", [])
                    ]
                elif source == "reddit":
                    return [
                        {
                            "title": post["data"]["title"],
                            "publishedAt": post["data"]["created_utc"],
                        }
                        for post in response.json()["data"]["children"]
                    ]
                elif source == "twitter":
                    return [
                        {"title": tweet["text"], "publishedAt": tweet["created_at"]}
                        for tweet in response.json().get("data", [])
                    ]

            except requests.exceptions.RequestException as e:
                logging.error(f"❌ Error fetching news for {ticker} from {source}: {e}")

        logging.warning(
            f"⚠ Max retries reached for {ticker} using {source}. Skipping..."
        )

    return []  # Return empty if all sources fail


### **🚀 Sentiment Analysis**
def analyze_sentiment(text):
    """
    Analyzes sentiment of a news article using TextBlob.
    Returns a polarity score between -1 (negative) to 1 (positive).
    """
    return round(TextBlob(text).sentiment.polarity, 3)


### **🚀 Store Sentiment Data in Database**
def store_sentiment_data(cursor, ticker, articles):
    """
    Stores news sentiment data in the database using batch insert.
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


### **🚀 Process News Sentiment for All Stocks**
def process_news_sentiment():
    """
    Fetches news articles, analyzes sentiment, and stores results in the database.
    Uses multiple news sources to avoid 429 errors.
    """
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT ticker FROM stocks;")
    tickers = [row[0] for row in cursor.fetchall()]

    console.print(
        f"\n[bold cyan]🚀 Fetching and analyzing news sentiment for {len(tickers)} stocks...[/bold cyan]\n"
    )

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
                logging.info(f"✅ Stored news sentiment for {ticker}")

            time.sleep(1)
            progress.update(task, advance=1)

    conn.commit()
    cursor.close()
    conn.close()
    console.print("[bold green]✅ News sentiment analysis completed![/bold green]")


### **🚀 Run the Script**
if __name__ == "__main__":
    process_news_sentiment()
