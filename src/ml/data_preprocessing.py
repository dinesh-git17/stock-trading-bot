import pandas as pd
import os
import logging
import warnings
from dotenv import load_dotenv
from rich.console import Console
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/data_preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# Directory Paths
TRAINING_DATA_PATH = "data/training_data_files/"
PROCESSED_DATA_PATH = "data/processed_data/"
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


def load_data(ticker):
    """Loads the training data for a given ticker."""
    file_path = os.path.join(TRAINING_DATA_PATH, f"training_data_{ticker}.csv")
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    return df if not df.empty else None


def handle_missing_data(df):
    """Handles missing values using forward fill and backward fill."""
    df = df.ffill().bfill()
    df.dropna(inplace=True)  # Drop rows if any NaNs remain
    return df if not df.empty else None  # Return None if empty


def feature_engineering(df):
    """Performs feature engineering by adding rolling averages and time-lagged features."""
    df = df.sort_values(by="date")

    # Moving Averages (SMA & EMA)
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # Volatility (Rolling Std Deviation)
    df["volatility"] = df["close"].rolling(window=20).std()

    # Time-Lagged Features (Previous Close Prices)
    for lag in range(1, 6):
        df[f"close_lag_{lag}"] = df["close"].shift(lag)

    df.dropna(inplace=True)
    return df if not df.empty else None  # Return None if empty


def scale_data(df):
    """Scales numerical features using MinMaxScaler, ensuring non-empty data."""
    feature_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_50",
        "sma_200",
        "ema_50",
        "ema_200",
        "volatility",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "sentiment_score",
    ] + [f"close_lag_{i}" for i in range(1, 6)]

    feature_columns = [col for col in feature_columns if col in df.columns]

    if df.empty or len(df) < 1:
        logging.warning("Skipping MinMaxScaler - No valid data available.")
        return None  # Skip if no valid data

    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df


def train_test_split_data(df, ticker):
    """Splits the data into train-test sets and saves them, ensuring non-empty data."""
    if df is None or df.empty:
        logging.warning(f"Skipping {ticker} - No data after preprocessing.")
        console.print(f"[bold red]⚠ Skipping {ticker} - No valid data.[/bold red]")
        return

    train, test = train_test_split(df, test_size=0.2, shuffle=False)

    train_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_train.csv")
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")

    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)

    logging.info(f"Saved train-test data for {ticker}")
    console.print(f"[bold green]✅ Processed & saved: {ticker}[/bold green]")


def process_ticker(ticker):
    """Processes a single ticker: loads, cleans, engineers features, scales, and splits data."""
    console.print(f"[bold yellow]⏳ Processing {ticker}...[/bold yellow]")

    df = load_data(ticker)
    if df is None:
        console.print(f"[bold red]⚠ No data for {ticker}. Skipping.[/bold red]")
        return

    df = handle_missing_data(df)
    if df is None:
        console.print(
            f"[bold red]⚠ {ticker} has no valid data after missing value handling. Skipping.[/bold red]"
        )
        return

    df = feature_engineering(df)
    if df is None:
        console.print(
            f"[bold red]⚠ {ticker} has no valid data after feature engineering. Skipping.[/bold red]"
        )
        return

    df = scale_data(df)
    if df is None:
        console.print(
            f"[bold red]⚠ {ticker} has no valid data after scaling. Skipping.[/bold red]"
        )
        return

    train_test_split_data(df, ticker)


def process_all_data():
    """Processes all tickers using parallel processing, ensuring only valid tickers are used."""
    tickers = [
        f.split("_")[-1].replace(".csv", "")
        for f in os.listdir(TRAINING_DATA_PATH)
        if f.startswith("training_data")
    ]

    console.print(
        f"[bold cyan]🚀 Processing {len(tickers)} tickers in parallel using {cpu_count()} cores...[/bold cyan]"
    )

    with Pool(cpu_count()) as pool:
        pool.map(process_ticker, tickers)

    console.print("[bold green]✅ All data preprocessing complete![/bold green]")


if __name__ == "__main__":
    process_all_data()
