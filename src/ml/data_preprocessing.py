import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from sklearn.preprocessing import MinMaxScaler

# ✅ Setup Logging
LOG_FILE = "data/logs/data_preprocessing.log"
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

# ✅ Directories
INPUT_DIR = "data/training_data"
OUTPUT_DIR = "data/transformed"
SCALER_DIR = "data/transformed/scalers"  # Store scalers properly
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# ✅ Configuration
LOOKBACK_DAYS = 10  # Number of past days used as features
NUM_WORKERS = min(4, os.cpu_count())  # Use max 4 cores to prevent overloading


def load_data(ticker):
    """Loads preprocessed stock data for a given ticker."""
    file_path = f"{INPUT_DIR}/training_data_{ticker}.csv"

    if not os.path.exists(file_path):
        logging.warning(f"⚠️ No data found for {ticker}. Skipping...")
        return None

    return pd.read_csv(file_path)


def transform_data(df):
    """Prepares data for LSTM model by creating lag features and normalizing."""

    # ✅ Ensure date is in datetime format & sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date")

    # ✅ Create Target Column (Next Day's Closing Price)
    df["next_close"] = df["close"].shift(-1)
    df.dropna(inplace=True)  # Remove last row with NaN target

    # ✅ Select Features
    features = [
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

    # ✅ Normalize Data
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # ✅ Create Lag Features (Time-Series Window)
    X, Y = [], []
    for i in range(len(df) - LOOKBACK_DAYS):
        X.append(
            df[features].iloc[i : i + LOOKBACK_DAYS].values
        )  # Past LOOKBACK_DAYS data
        Y.append(df["next_close"].iloc[i + LOOKBACK_DAYS])  # Next closing price

    X, Y = np.array(X), np.array(Y)  # Convert to numpy arrays

    return X, Y, scaler


def save_transformed_data(ticker, X, Y, scaler):
    """Saves transformed dataset & scaler for each ticker."""
    output_file = f"{OUTPUT_DIR}/transformed_{ticker}.pkl"
    scaler_file = f"{SCALER_DIR}/scaler_{ticker}.pkl"

    with open(output_file, "wb") as f:
        pickle.dump((X, Y), f)

    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)

    logging.info(f"✅ Transformed data saved for {ticker}: {output_file}")


def process_ticker(ticker):
    """Processes a single ticker: Loads, Transforms, and Saves."""
    df = load_data(ticker)
    if df is None:
        return ticker, False  # Indicate failure

    X, Y, scaler = transform_data(df)
    save_transformed_data(ticker, X, Y, scaler)

    return ticker, True  # Indicate success


if __name__ == "__main__":
    console.print(
        "[bold cyan]🚀 Processing stock data for LSTM training using multiprocessing...[/bold cyan]"
    )

    tickers = [
        f.split("_")[-1].split(".")[0]
        for f in os.listdir(INPUT_DIR)
        if f.startswith("training_data_")
    ]

    # ✅ Rich Progress Bar for Processing (Ensuring It Prints Only Once)
    with Progress(
        TextColumn("[bold blue]⏳ Transforming Data:[/bold blue] {task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing Tickers...", total=len(tickers))

        # ✅ Use multiprocessing to process tickers in parallel
        results = []
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_ticker = {
                executor.submit(process_ticker, ticker): ticker for ticker in tickers
            }

            for future in as_completed(future_to_ticker):
                ticker, success = future.result()
                results.append((ticker, success))
                progress.update(
                    task, advance=1
                )  # ✅ Updates bar dynamically WITHOUT multiple prints

    # ✅ Print all success/failure messages AFTER the progress bar is complete
    for ticker, success in results:
        if success:
            console.print(
                f"[bold green]✅ Saved transformed data for {ticker}[/bold green]"
            )
        else:
            console.print(f"[bold red]⚠️ Skipped {ticker} (No Data)[/bold red]")

    console.print(
        "\n[bold green]✅ Data transformation complete! All tickers processed successfully.[/bold green]\n"
    )
