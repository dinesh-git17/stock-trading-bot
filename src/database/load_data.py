import logging
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/database_load.log",
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

CLEANED_DATA_DIR = "data/cleaned/"


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


def insert_stock_data(cursor, ticker, df):
    """
    Inserts OHLCV data into the `stocks` table.
    """
    sql = """
        INSERT INTO stocks (ticker, date, open, high, low, close, volume, dividends, stock_splits)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, date) DO NOTHING;
    """
    records = [
        (
            ticker,
            row.Index,
            getattr(row, "Open", None),
            getattr(row, "High", None),
            getattr(row, "Low", None),
            getattr(row, "Close", None),
            getattr(row, "Volume", None),
            getattr(row, "Dividends", None),
            getattr(row, "Stock Splits", None),
        )
        for row in df.itertuples()
    ]
    cursor.executemany(sql, records)


def insert_technical_indicators(cursor, ticker, df):
    """
    Inserts technical indicator data into the `technical_indicators` table.
    """
    sql = """
        INSERT INTO technical_indicators (ticker, date, sma_20, ema_20, rsi_14, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, date) DO NOTHING;
    """
    records = [
        (
            ticker,
            row.Index,
            getattr(row, "SMA_20", None),
            getattr(row, "EMA_20", None),
            getattr(row, "RSI_14", None),
            getattr(row, "MACD", None),
            getattr(row, "MACD_Signal", None),
            getattr(row, "MACD_Hist", None),
            getattr(row, "BB_Upper", None),
            getattr(row, "BB_Middle", None),
            getattr(row, "BB_Lower", None),
        )
        for row in df.itertuples()
    ]
    cursor.executemany(sql, records)


def load_data():
    """
    Reads cleaned stock data from `data/cleaned/` and loads it into the database.
    """
    console.print("\n")
    files = [f for f in os.listdir(CLEANED_DATA_DIR) if f.endswith("_cleaned.csv")]

    conn = connect_db()
    if not conn:
        return
    cursor = conn.cursor()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Loading stock data into database...[/bold yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("loading", total=len(files))

        for file in files:
            ticker = file.replace("_cleaned.csv", "")
            file_path = os.path.join(CLEANED_DATA_DIR, file)
            df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

            if df.empty:
                logging.warning(f"No data to load for {ticker}")
                continue

            try:
                insert_stock_data(cursor, ticker, df)
                insert_technical_indicators(cursor, ticker, df)
                conn.commit()
                logging.info(f"Successfully loaded data for {ticker}")

            except Exception as e:
                logging.error(f"Failed to load data for {ticker}: {e}")
                conn.rollback()

            progress.update(task, advance=1)

    cursor.close()
    conn.close()
    console.print(
        "[bold green]✅ Data successfully loaded into the database![/bold green]\n"
    )


if __name__ == "__main__":
    load_data()
