import concurrent.futures
import logging
import os

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from scipy.stats import zscore

# ✅ Import utilities
from src.tools.utils import handle_exceptions, setup_logging

# ✅ Setup Logging
LOG_FILE = "data/logs/data_cleaning.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)
console = Console()
logger.info("🚀 Logging setup complete.")

# ✅ Directories
PROCESSED_DATA_DIR = "data/processed/"
CLEANED_DATA_DIR = "data/cleaned/"
os.makedirs(CLEANED_DATA_DIR, exist_ok=True)

# ✅ Parallel processing threads
THREADS = 4


@handle_exceptions
def handle_missing_values(df):
    """Handles missing values using forward fill, backward fill, and interpolation."""
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.interpolate(method="linear", inplace=True)
    return df


@handle_exceptions
def remove_outliers(df, column="close", threshold=3.0):
    """Removes outliers using Z-score method."""
    if column in df.columns:
        df = df.copy()
        df["zscore"] = zscore(df[column].dropna())
        df = df[df["zscore"].abs() < threshold].copy()
        df.drop(columns=["zscore"], inplace=True)
    return df


@handle_exceptions
def clean_stock_data(ticker):
    """Cleans stock data by handling missing values and removing outliers."""
    processed_file = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")

    if not os.path.exists(processed_file):
        logger.warning(f"⚠ Processed data file missing for {ticker}. Skipping...")
        return

    df = pd.read_csv(processed_file)
    if df.empty:
        logger.warning(f"⚠ No data found in {processed_file}. Skipping...")
        return

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df.set_index("Date", inplace=True)
    else:
        logger.warning(f"⚠ No valid 'Date' column found in {ticker}. Skipping...")
        return

    df.columns = df.columns.str.lower()
    df = handle_missing_values(df)
    df = remove_outliers(df, column="close")

    cleaned_file = os.path.join(CLEANED_DATA_DIR, f"{ticker}_cleaned.csv")
    df.to_csv(cleaned_file)
    logger.info(f"✅ Saved cleaned data for {ticker} to {cleaned_file}")


@handle_exceptions
def process_all_stocks():
    """Cleans all available stock data using multi-threading."""
    tickers = [
        file.replace("_processed.csv", "")
        for file in os.listdir(PROCESSED_DATA_DIR)
        if file.endswith("_processed.csv")
    ]
    console.print(
        Panel(f"🚀 Cleaning data for {len(tickers)} stocks...", style="bold cyan")
    )

    with Progress(
        TextColumn("[bold yellow]Processing:[/bold yellow]"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing", total=len(tickers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = {
                executor.submit(clean_stock_data, ticker): ticker for ticker in tickers
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                progress.update(task, advance=1)

    console.print(Panel("✅ Done cleaning stock data!", style="bold green"))


if __name__ == "__main__":
    process_all_stocks()
