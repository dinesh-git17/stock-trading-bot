import logging
import os

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from scipy.stats import zscore

# Setup logging
logging.basicConfig(
    filename="data/logs/data_cleaning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

PROCESSED_DATA_DIR = "data/processed/"
CLEANED_DATA_DIR = "data/cleaned/"
os.makedirs(CLEANED_DATA_DIR, exist_ok=True)  # Ensure cleaned data directory exists


def handle_missing_values(df):
    """
    Handles missing values using forward fill and interpolation.
    """
    df.ffill(inplace=True)  # ✅ Fix: Use df.ffill() instead of fillna(method="ffill")
    df.interpolate(method="linear", inplace=True)  # Linear interpolation
    return df


def remove_outliers(df, column="Close", threshold=3.0):
    """
    Removes outliers using Z-score method.
    """
    if column in df.columns:
        df["zscore"] = zscore(df[column].dropna())
        df = df[df["zscore"].abs() < threshold]  # Keep only within threshold
        df = df.drop(columns=["zscore"])  # ✅ Fix: Assign df after drop()
    return df


def clean_stock_data(ticker):
    """
    Cleans stock data by handling missing values and removing outliers.
    Saves cleaned data to data/cleaned/.
    """
    processed_file = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")

    if not os.path.exists(processed_file):
        logging.warning(f"Processed data file missing for {ticker}")
        return  # Skip if processed data does not exist

    df = pd.read_csv(processed_file, index_col="Date", parse_dates=True)

    if df.empty:
        logging.warning(f"No data found in {processed_file}")
        return

    # Handle missing values
    df = handle_missing_values(df)

    # Remove outliers
    df = remove_outliers(df, column="Close")

    # Save cleaned data
    cleaned_file = os.path.join(CLEANED_DATA_DIR, f"{ticker}_cleaned.csv")
    df.to_csv(cleaned_file)

    logging.info(f"Saved cleaned data for {ticker} to {cleaned_file}")


def process_all_stocks():
    """
    Cleans all available stock data in data/processed/.
    """
    console.print("\n")
    files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith("_processed.csv")]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Cleaning Stock Data...[/bold yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("cleaning", total=len(files))

        for file in files:
            ticker = file.replace("_processed.csv", "")
            clean_stock_data(ticker)
            progress.update(task, advance=1)

    console.print("[bold green]✅ Done cleaning stock data![/bold green]\n")


if __name__ == "__main__":
    process_all_stocks()
