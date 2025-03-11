import concurrent.futures
import logging
import os
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ✅ Load environment variables
load_dotenv()

# ✅ Setup logging
logging.basicConfig(
    filename="data/logs/data_preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ Directory Paths
TRAINING_DATA_PATH = "data/training_data_files/"
PROCESSED_DATA_PATH = "data/processed_data/"
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


def load_data(ticker):
    """Loads the training data for a given ticker."""
    file_path = os.path.join(TRAINING_DATA_PATH, f"training_data_{ticker}.csv")
    if not os.path.exists(file_path):
        logging.warning(f"⚠ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    return df if not df.empty else None


def handle_missing_data(df):
    """Handles missing values using interpolation and fills."""
    if df.isnull().sum().sum() > 0:
        df.interpolate(method="linear", inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    df.dropna(inplace=True)

    if df.empty:
        logging.warning("⚠ All data removed after missing value handling!")
        return None

    return df


def feature_engineering(df):
    """Performs feature engineering by adding rolling averages and time-lagged features."""
    df = df.sort_values(by="date")

    # ✅ Moving Averages (SMA & EMA)
    df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # ✅ Volatility (Rolling Std Deviation)
    df["volatility"] = df["close"].rolling(window=10, min_periods=1).std()

    # ✅ Time-Lagged Features (Previous Close Prices)
    for lag in range(1, 6):
        df[f"close_lag_{lag}"] = df["close"].shift(lag)

    df.dropna(inplace=True)

    if df.empty:
        logging.warning("⚠ Data removed after feature engineering!")
        return None

    return df


def scale_data(df):
    """Scales numerical features using StandardScaler."""
    feature_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_20",
        "ema_20",
        "volatility",
    ] + [f"close_lag_{i}" for i in range(1, 6)]

    feature_columns = [col for col in feature_columns if col in df.columns]

    if df.empty or len(df) < 1:
        logging.warning("⚠ Skipping StandardScaler - No valid data available.")
        return None

    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df


def train_test_split_data(df, ticker):
    """Splits the data into train-test sets and saves them."""
    if df is None or df.empty:
        logging.warning(f"⚠ Skipping {ticker} - No data after preprocessing.")
        return

    train, test = train_test_split(df, test_size=0.2, shuffle=False)

    train.to_csv(os.path.join(PROCESSED_DATA_PATH, f"{ticker}_train.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv"), index=False)

    logging.info(f"✅ Saved train-test data for {ticker}")


def process_ticker(ticker):
    """Processes a single ticker: loads, cleans, engineers features, scales, and splits data."""
    try:
        df = load_data(ticker)
        if df is None:
            return f"⚠ No data for {ticker}, Skipping."

        df = handle_missing_data(df)
        if df is None:
            return f"⚠ {ticker} empty after missing data handling."

        df = feature_engineering(df)
        if df is None:
            return f"⚠ {ticker} empty after feature engineering."

        df = scale_data(df)
        if df is None:
            return f"⚠ {ticker} empty after scaling."

        train_test_split_data(df, ticker)
        return f"✅ {ticker} processed & saved!"

    except Exception as e:
        logging.error(f"❌ Error processing {ticker}: {str(e)}")
        return f"❌ {ticker} processing failed!"


def process_all_data():
    """Processes all tickers using parallel processing with structured progress reporting."""
    tickers = [
        f.split("_")[-1].replace(".csv", "")
        for f in os.listdir(TRAINING_DATA_PATH)
        if f.startswith("training_data")
    ]

    console.print(
        f"[bold cyan]🚀 Processing {len(tickers)} tickers using {min(8, os.cpu_count())} threads...[/bold cyan]"
    )

    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, os.cpu_count())
    ) as executor:
        for result in executor.map(process_ticker, tickers):
            results.append(result)

    console.print("\n[bold green]✅ All data preprocessing complete![/bold green]\n")

    # ✅ Display summary table
    table = Table(title="Preprocessing Summary")
    table.add_column("Status", justify="center", style="cyan", no_wrap=True)

    for result in results:
        table.add_row(result)

    console.print(table)


if __name__ == "__main__":
    process_all_data()
