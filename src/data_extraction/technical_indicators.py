import logging
import os

import pandas as pd
import pandas_ta as ta
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup logging
logging.basicConfig(
    filename="data/logs/technical_indicators.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)  # Ensure processed data directory exists


def compute_technical_indicators(ticker):
    """
    Compute SMA, EMA, RSI, MACD, and Bollinger Bands for a given stock ticker.
    Saves processed data into data/processed/.
    """
    raw_file = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")

    if not os.path.exists(raw_file):
        logging.warning(f"Raw data file missing for {ticker}")
        return  # Skip if raw data does not exist

    df = pd.read_csv(raw_file, index_col="Date", parse_dates=True)

    if df.empty:
        logging.warning(f"No data found in {raw_file}")
        return

    # Compute Technical Indicators
    df["SMA_20"] = ta.sma(df["Close"], length=20)  # 20-day Simple Moving Average
    df["EMA_20"] = ta.ema(df["Close"], length=20)  # 20-day Exponential Moving Average
    df["RSI_14"] = ta.rsi(df["Close"], length=14)  # 14-day Relative Strength Index
    macd = ta.macd(df["Close"])  # MACD
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = (
        macd["MACD_12_26_9"],
        macd["MACDs_12_26_9"],
        macd["MACDh_12_26_9"],
    )
    bbands = ta.bbands(df["Close"], length=20)  # Bollinger Bands
    df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = (
        bbands["BBU_20_2.0"],
        bbands["BBM_20_2.0"],
        bbands["BBL_20_2.0"],
    )

    # Save processed data
    processed_file = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
    df.to_csv(processed_file)

    logging.info(f"Saved processed data for {ticker} to {processed_file}")


def process_all_stocks():
    """
    Compute technical indicators for all available stock data in data/raw/.
    """
    console.print("\n")
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Computing Technical Indicators...[/bold yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("processing", total=len(files))

        for file in files:
            ticker = file.replace(".csv", "")
            compute_technical_indicators(ticker)
            progress.update(task, advance=1)

    console.print("[bold green]✅ Done computing technical indicators![/bold green]\n")


if __name__ == "__main__":
    process_all_stocks()
