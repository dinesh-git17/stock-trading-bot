import pandas as pd
import psycopg2
import os
import logging
import signal
import warnings
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from multiprocessing import Pool, cpu_count

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


# Handle Ctrl+C to exit safely
def signal_handler(sig, frame):
    console.print("\n[bold red]❌ Process interrupted. Exiting safely...[/bold red]")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def connect_db():
    """Connects to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        console.print(f"[bold red]❌ Database connection error:[/bold red] {e}")
        logging.error(f"Database connection error: {e}")
        return None


def fetch_all_stock_data():
    """Fetches all stock & technical indicator data in one query to optimize performance."""
    conn = connect_db()
    if not conn:
        return pd.DataFrame()

    query = """
        SELECT s.ticker, s.date, s.open, s.high, s.low, s.close, s.volume, 
               ti.sma_20, ti.ema_20, ti.rsi_14, ti.macd, ti.macd_signal, 
               ti.macd_hist, ti.bb_upper, ti.bb_middle, ti.bb_lower
        FROM stock_analysis_view s
        LEFT JOIN technical_indicators ti ON s.ticker = ti.ticker AND s.date = ti.date
        ORDER BY s.ticker, s.date;
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df


def fetch_all_sentiment_data():
    """Fetches all sentiment data in one batch query."""
    conn = connect_db()
    if not conn:
        return pd.DataFrame()

    query = """
        SELECT ticker, published_at, sentiment_score
        FROM news_sentiment
        ORDER BY ticker, published_at;
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df


def get_tickers_from_db():
    """Fetch all tickers from the database."""
    conn = connect_db()
    if not conn:
        return []

    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM stocks;")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()

    return tickers


def merge_data(df_stock, df_sentiment):
    """Merges stock price data with sentiment scores for a given ticker."""
    df_sentiment["date"] = pd.to_datetime(df_sentiment["published_at"]).dt.date
    df_stock["date"] = pd.to_datetime(df_stock["date"]).dt.date

    df_merged = pd.merge(df_stock, df_sentiment, on="date", how="left")
    df_merged = df_merged.ffill().bfill()  # Fill forward & backward for missing data

    return df_merged


def save_training_data(args):
    """Fetches and saves the training data for a single ticker."""
    ticker, all_stock_data, all_sentiment_data = args

    console.print(f"[bold yellow]⏳ Fetching data for {ticker}...[/bold yellow]")

    df_stock = all_stock_data[all_stock_data["ticker"] == ticker].drop(
        columns=["ticker"]
    )
    df_sentiment = all_sentiment_data[all_sentiment_data["ticker"] == ticker].drop(
        columns=["ticker"]
    )

    if df_stock.empty:
        logging.warning(f"No stock data found for {ticker}")
        console.print(f"[bold red]⚠ No stock data for {ticker}! Skipping.[/bold red]")
        return

    if df_sentiment.empty:
        logging.warning(f"No sentiment data found for {ticker}")
        df_sentiment["date"] = df_stock["date"]
        df_sentiment["sentiment_score"] = 0  # Neutral sentiment if no data

    df_merged = merge_data(df_stock, df_sentiment)

    output_folder = "data/training_data_files/"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"training_data_{ticker}.csv")
    df_merged.to_csv(output_file, index=False)

    logging.info(f"Saved training data for {ticker} to {output_file}")
    console.print(f"[bold green]✅ {ticker} saved successfully![/bold green]")


def process_training_data():
    """Fetches and processes data for all tickers in parallel."""
    console.print("[bold blue]📊 Fetching all stock & sentiment data...[/bold blue]")

    all_stock_data = fetch_all_stock_data()
    all_sentiment_data = fetch_all_sentiment_data()

    if all_stock_data.empty or all_sentiment_data.empty:
        console.print(
            "[bold red]⚠ No data fetched. Check database connection.[/bold red]"
        )
        return

    tickers = get_tickers_from_db()
    if not tickers:
        console.print("[bold red]⚠ No tickers found in the database![/bold red]")
        return

    console.print(
        f"[bold cyan]🚀 Processing {len(tickers)} tickers in parallel using {cpu_count()} cores...[/bold cyan]"
    )

    args = [(ticker, all_stock_data, all_sentiment_data) for ticker in tickers]

    with Pool(cpu_count()) as pool:
        pool.map(save_training_data, args)

    console.print("[bold green]✅ All training data processing complete![/bold green]")


if __name__ == "__main__":
    process_training_data()
