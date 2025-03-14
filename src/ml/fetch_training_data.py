import logging
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/fetch_training_data.log"
os.makedirs("data/logs", exist_ok=True)

# ✅ Insert blank lines before logging new logs
with open(LOG_FILE, "a") as log_file:
    log_file.write("\n" * 3)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ Read database credentials from environment variables
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# ✅ Validate that environment variables are set
if None in [DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT]:
    console.print(
        "[bold red]❌ ERROR:[/bold red] Missing database environment variables.",
        style="bold red",
    )
    exit(1)

# ✅ Output directory for processed training data
OUTPUT_DIR = "data/training_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_data():
    """Fetches OHLCV stock data from `stocks` and technical indicators from `technical_indicators`."""
    try:
        engine = create_engine(
            f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        conn = engine.connect()

        # ✅ Query stock OHLCV data from `stocks`
        stocks_query = """
        SELECT ticker, date, open, high, low, close, volume, adjusted_close
        FROM stocks;
        """

        # ✅ Query technical indicators (Using Correct Columns)
        indicators_query = """
        SELECT ticker, date, sma_50, sma_200, ema_50, ema_200, rsi_14, adx_14, atr_14, cci_20, williamsr_14, 
               macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower, stoch_k, stoch_d
        FROM technical_indicators;
        """

        # ✅ Load data into Pandas DataFrames
        stocks_df = pd.read_sql(stocks_query, conn)
        indicators_df = pd.read_sql(indicators_query, conn)

        conn.close()
        return stocks_df, indicators_df

    except Exception as e:
        logging.error(f"Database fetch error: {e}")
        console.print(
            f"[bold red]❌ ERROR:[/bold red] Failed to fetch data from database.",
            style="bold red",
        )
        return None, None


def preprocess_and_save_data(stocks_df, indicators_df):
    """Processes and saves training data for each ticker separately."""

    # ✅ Merge stock data with technical indicators
    df = pd.merge(stocks_df, indicators_df, on=["ticker", "date"], how="left")

    # ✅ Convert date to datetime format
    df["date"] = pd.to_datetime(df["date"])

    # ✅ Sort by ticker and date
    df = df.sort_values(by=["ticker", "date"])

    # ✅ Handle missing values (FIXED FUTURE WARNING)
    df = df.ffill().bfill()

    # ✅ Normalize numerical data using MinMaxScaler
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted_close",
        "sma_50",
        "sma_200",
        "ema_50",
        "ema_200",
        "rsi_14",
        "adx_14",
        "atr_14",
        "cci_20",
        "williamsr_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "stoch_k",
        "stoch_d",
    ]

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # ✅ Get unique tickers
    tickers = df["ticker"].unique()

    console.print(
        "\n[bold cyan]📊 Saving processed data for each ticker...[/bold cyan]"
    )

    # ✅ Rich progress bar for saving files
    with Progress(
        TextColumn("[bold blue]⏳ Processing[/bold blue] {task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Saving Data...", total=len(tickers))

        for ticker in tickers:
            ticker_df = df[df["ticker"] == ticker]

            # ✅ Save each ticker's data in an industry-level format
            filename = f"{OUTPUT_DIR}/training_data_{ticker}.csv"
            ticker_df.to_csv(filename, index=False)

            logging.info(f"Saved processed data for {ticker} -> {filename}")
            progress.update(task, advance=1)

    console.print("[bold green]✅ Data preprocessing complete![/bold green]")


if __name__ == "__main__":
    console.print("[bold cyan]🚀 Fetching stock data...[/bold cyan]")
    stocks_df, indicators_df = fetch_data()

    if stocks_df is not None and indicators_df is not None:
        preprocess_and_save_data(stocks_df, indicators_df)
    else:
        console.print("[bold red]❌ Process aborted due to database error.[/bold red]")
