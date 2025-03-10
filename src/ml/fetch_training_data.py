import logging
import os
import signal
import sys
import warnings
from multiprocessing import Pool, cpu_count

import pandas as pd
import psycopg2
from alive_progress import alive_bar
from dotenv import load_dotenv
from psycopg2 import pool
from rich.console import Console
from rich.table import Table

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/training_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Database Configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

# Create a connection pool for efficient DB access
db_pool = pool.SimpleConnectionPool(
    1, 10, **DB_CONFIG
)  # Min 1, Max 10 connections in the pool


# Handle Ctrl+C to exit safely
def signal_handler(sig, frame):
    console.print("\n[bold red]❌ Process interrupted. Exiting safely...[/bold red]")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def get_db_connection():
    """Fetches a connection from the pool."""
    try:
        return db_pool.getconn()
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return None


def fetch_data(query):
    """Fetch data from the database using a connection pool."""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
        return pd.DataFrame(data, columns=columns)
    except Exception as e:
        logging.error(f"Database query error: {e}")
        return pd.DataFrame()
    finally:
        db_pool.putconn(conn)


def fetch_all_stock_data():
    """Fetches all stock & technical indicator data in one query."""
    query = """
        SELECT s.ticker, s.date, s.open, s.high, s.low, s.close, s.volume, 
               ti.sma_20, ti.ema_20, ti.rsi_14, ti.macd, ti.macd_signal, 
               ti.macd_hist, ti.bb_upper, ti.bb_middle, ti.bb_lower
        FROM stock_analysis_view s
        LEFT JOIN technical_indicators ti ON s.ticker = ti.ticker AND s.date = ti.date
        ORDER BY s.ticker, s.date;
    """
    return fetch_data(query)


def fetch_all_sentiment_data():
    """Fetches all sentiment data, ensuring an empty dataset does not break merging."""
    query = """
        SELECT ticker, published_at, sentiment_score
        FROM news_sentiment
        ORDER BY ticker, published_at;
    """
    df = fetch_data(query)

    if df.empty:
        logging.warning("No sentiment data found. Defaulting to neutral sentiment.")
        return pd.DataFrame({"ticker": [], "published_at": [], "sentiment_score": []})

    return df


def get_tickers_from_db():
    """Fetch all distinct tickers from the database."""
    query = "SELECT DISTINCT ticker FROM stocks;"
    df = fetch_data(query)
    return df["ticker"].tolist() if not df.empty else []


def merge_data(df_stock, df_sentiment):
    """Merges stock price data with sentiment scores and fills missing values with neutral sentiment."""
    df_stock["date"] = pd.to_datetime(df_stock["date"]).dt.date

    if df_sentiment.empty or "published_at" not in df_sentiment.columns:
        df_sentiment = pd.DataFrame(
            {
                "date": df_stock["date"],
                "published_at": df_stock["date"],
                "sentiment_score": 0,
            }
        )

    df_sentiment["date"] = pd.to_datetime(df_sentiment["published_at"]).dt.date

    df_merged = pd.merge(df_stock, df_sentiment, on="date", how="left").fillna(0)

    return df_merged


def save_training_data(args):
    """Fetches and saves the training data for a single ticker."""
    ticker, all_stock_data, all_sentiment_data = args

    df_stock = all_stock_data[all_stock_data["ticker"] == ticker].drop(
        columns=["ticker"]
    )
    df_sentiment = all_sentiment_data[all_sentiment_data["ticker"] == ticker].drop(
        columns=["ticker"]
    )

    if df_stock.empty:
        logging.warning(f"No stock data for {ticker}")
        return f"⚠ No stock data for {ticker}! Skipping."

    if df_sentiment.empty:
        df_sentiment = pd.DataFrame({"date": df_stock["date"], "sentiment_score": 0})

    df_merged = merge_data(df_stock, df_sentiment)

    output_folder = "data/training_data_files/"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"training_data_{ticker}.csv")
    df_merged.to_csv(output_file, index=False)

    logging.info(f"Saved training data for {ticker} to {output_file}")
    return f"✅ {ticker} saved successfully!"


def process_training_data():
    """Fetches and processes data for all tickers in parallel with a modern progress bar."""
    console.print("[bold blue]📊 Fetching all stock & sentiment data...[/bold blue]")

    all_stock_data = fetch_all_stock_data()
    all_sentiment_data = fetch_all_sentiment_data()

    if all_stock_data.empty:
        console.print(
            "[bold red]⚠ No stock data found! Check database connection.[/bold red]"
        )
        return

    tickers = get_tickers_from_db()
    if not tickers:
        console.print("[bold red]⚠ No tickers found in the database![/bold red]")
        return

    console.print(
        f"[bold cyan]🚀 Processing {len(tickers)} tickers using {min(8, cpu_count())} cores...[/bold cyan]"
    )

    args = [(ticker, all_stock_data, all_sentiment_data) for ticker in tickers]

    # Use alive-progress for a more modern progress bar
    results = []
    with Pool(min(8, cpu_count())) as pool, alive_bar(
        len(tickers), title="Processing Tickers", bar="smooth"
    ) as bar:
        for res in pool.imap(save_training_data, args):
            results.append(res)
            bar()  # Update progress bar

    console.print(
        "\n[bold green]✅ All training data processing complete![/bold green]\n"
    )

    # Display summary table
    table = Table(title="Training Data Processing Summary")
    table.add_column("Status", justify="center", style="cyan", no_wrap=True)

    for result in results:
        table.add_row(result)

    console.print(table)


if __name__ == "__main__":
    process_training_data()
