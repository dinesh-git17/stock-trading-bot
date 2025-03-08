import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import warnings

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/training_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# Suppress any warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Database Configuration
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")


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


def fetch_data_from_db(ticker):
    """Fetches stock price data (OHLCV) and technical indicators from the database."""
    conn = connect_db()
    if not conn:
        return None

    # Query to fetch stock price data (OHLCV) and technical indicators
    query = f"""
        SELECT s.date, s.open, s.high, s.low, s.close, s.volume, 
               ti.sma_20, ti.ema_20, ti.rsi_14, ti.macd, ti.macd_signal, 
               ti.macd_hist, ti.bb_upper, ti.bb_middle, ti.bb_lower
        FROM stock_analysis_view s
        LEFT JOIN technical_indicators ti ON s.ticker = ti.ticker AND s.date = ti.date
        WHERE s.ticker = '{ticker}'
        ORDER BY s.date;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def fetch_sentiment_data(ticker):
    """Fetches sentiment data for the given ticker from the database."""
    conn = connect_db()
    if not conn:
        return None

    query = f"""
        SELECT published_at, sentiment_score
        FROM news_sentiment
        WHERE ticker = '{ticker}'
        ORDER BY published_at;
    """

    df_sentiment = pd.read_sql(query, conn)
    conn.close()
    return df_sentiment


def merge_data(df_stock, df_sentiment):
    """Merges stock price data with sentiment scores."""
    df_sentiment["date"] = pd.to_datetime(df_sentiment["published_at"]).dt.date
    df_stock["date"] = pd.to_datetime(df_stock["date"]).dt.date

    df_merged = pd.merge(df_stock, df_sentiment, on="date", how="left")
    return df_merged


def save_training_data(ticker):
    """Fetches and saves the training data for the given ticker."""
    console.print(f"\n[bold yellow]Fetching data for {ticker}...[/bold yellow]")

    df_stock = fetch_data_from_db(ticker)
    if df_stock is None or df_stock.empty:
        logging.warning(f"No stock data found for {ticker}")
        return

    df_sentiment = fetch_sentiment_data(ticker)
    if df_sentiment is None or df_sentiment.empty:
        logging.warning(f"No sentiment data found for {ticker}")
        return

    # Merge stock data with sentiment
    df_merged = merge_data(df_stock, df_sentiment)

    # Save to CSV in the training_data_files folder
    output_folder = "data/training_data_files/"
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
    output_file = os.path.join(output_folder, f"training_data_{ticker}.csv")
    df_merged.to_csv(output_file, index=False)
    logging.info(f"Saved training data for {ticker} to {output_file}")
    console.print(f"[bold green]✅ Data for {ticker} saved successfully![/bold green]")


def process_training_data():
    """Fetches and processes data for all tickers in the database."""
    # Fetch all tickers from the stocks table
    conn = connect_db()
    if not conn:
        return
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT ticker FROM stocks;")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Fetching & processing training data...[/bold yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("sentiment", total=len(tickers))

        for ticker in tickers:
            save_training_data(ticker)
            progress.update(task, advance=1)

    console.print("[bold green]✅ Training data processing complete![/bold green]")


if __name__ == "__main__":
    process_training_data()
