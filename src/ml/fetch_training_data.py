import logging
import os
import signal
import sys
import warnings
from multiprocessing import Pool, cpu_count

import pandas as pd
from dotenv import load_dotenv
from psycopg2 import pool
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/training_data.log"
os.makedirs("data/logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ✅ Database Configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

# ✅ Connection Pool
db_pool = pool.SimpleConnectionPool(1, 10, **DB_CONFIG)  # Min 1, Max 10 connections


# ✅ Graceful Exit Handler
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
    """Fetches all stock & technical indicator data from materialized views."""
    query = """
        SELECT * FROM materialized_stock_analysis
        ORDER BY ticker, date;
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

    df_merged = pd.merge(df_stock, df_sentiment, on="date", how="left").infer_objects(
        copy=False
    )

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
    console.print("\n[bold blue]📊 Fetching all stock & sentiment data...[/bold blue]")

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

    results = []
    with Pool(min(8, cpu_count())) as pool, Progress(
        SpinnerColumn(),
        TextColumn("[bold blue] Processing {task.fields[ticker]}..."),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(tickers), ticker="Starting...")

        for res in pool.imap(save_training_data, args):
            results.append(res)
            progress.update(
                task, advance=1, ticker=res.split()[1] if "✅" in res else "Skipping..."
            )

    console.print(
        "\n[bold green]✅ All training data processing complete![/bold green]\n"
    )

    # Display summary table with multiple tickers per row
    table = Table(
        title="Training Data Processing Summary", show_lines=True, expand=True
    )
    table.add_column(
        "Processed Tickers", justify="center", style="cyan", overflow="fold"
    )

    max_cols = 5  # Number of tickers per row
    grouped_results = [
        results[i : i + max_cols] for i in range(0, len(results), max_cols)
    ]

    for group in grouped_results:
        table.add_row(" | ".join(group))

    console.print(table)


if __name__ == "__main__":
    process_training_data()
