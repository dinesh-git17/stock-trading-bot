import concurrent.futures
import logging
import os
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/data_preprocessing.log"
os.makedirs("data/logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
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

    df = pd.read_csv(
        file_path,
        parse_dates=["date"],
        dtype={
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "volume": "int64",
        },
    )

    return df if not df.empty else None


def handle_missing_data(df):
    """Handles missing values using adaptive imputation."""
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Fill missing 'close' & 'volume' with median values
    df["close"].fillna(df["close"].median(), inplace=True)
    df["volume"].fillna(df["volume"].median(), inplace=True)

    df.dropna(subset=["close", "volume"], inplace=True)

    return df if not df.empty else None


def feature_engineering(df):
    """Adds technical indicators while ensuring minimal data loss."""
    df = df.sort_values(by="date")

    # ✅ Moving Averages
    df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["sma_50"] = df["close"].rolling(window=50, min_periods=1).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # ✅ Bollinger Bands
    df["volatility"] = df["close"].rolling(window=10, min_periods=1).std()
    df["bb_upper"] = df["sma_20"] + (df["volatility"] * 2)
    df["bb_lower"] = df["sma_20"] - (df["volatility"] * 2)

    # ✅ MACD (Moving Average Convergence Divergence)
    short_ema = df["close"].ewm(span=12, adjust=False).mean()
    long_ema = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = short_ema - long_ema
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # ✅ RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)  # Avoid division by zero
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ✅ Log Transform Volume (reduces outliers)
    df["log_volume"] = np.log1p(df["volume"])

    # ✅ Time-Lagged Features (Previous Prices)
    for lag in range(1, 6):
        df[f"close_lag_{lag}"] = df["close"].shift(lag)

    df.dropna(inplace=True)

    return df if not df.empty else None


def scale_data(df):
    """Scales numerical features using RobustScaler to handle outliers."""
    feature_columns = [
        "open",
        "high",
        "low",
        "close",
        "sma_20",
        "sma_50",
        "ema_20",
        "ema_50",
        "volatility",
        "bb_upper",
        "bb_lower",
        "macd",
        "macd_signal",
        "rsi_14",
        "log_volume",
    ] + [f"close_lag_{i}" for i in range(1, 6)]

    feature_columns = [col for col in feature_columns if col in df.columns]

    if df.empty:
        return None

    scaler = RobustScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df


def train_test_split_data(df, ticker):
    """Splits the data into train-test sets and saves them."""
    if df is None:
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

    console.print(f"[bold cyan]🚀 Processing {len(tickers)} tickers...[/bold cyan]")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, os.cpu_count())
    ) as executor:
        results = list(executor.map(process_ticker, tickers))

    console.print("\n[bold green]✅ All data preprocessing complete![/bold green]\n")

    table = Table(title="Preprocessing Summary")
    table.add_column("Status", justify="center", style="cyan", no_wrap=True)
    for result in results:
        table.add_row(result)

    console.print(table)


if __name__ == "__main__":
    process_all_data()
