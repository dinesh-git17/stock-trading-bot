import concurrent.futures
import logging
import os

import pandas as pd
import pandas_ta as ta
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

# ✅ Import utilities
from src.tools.utils import handle_exceptions, setup_logging

# ✅ Setup Logging
LOG_FILE = "data/logs/technical_indicators.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)
console = Console()
logger.info("🚀 Logging setup complete.")

# ✅ Directories
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ✅ Constants
THREADS = 4
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume", "adjusted_close"]


@handle_exceptions
def read_stock_data(ticker):
    """Reads stock data and ensures it has the necessary structure."""
    raw_file = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(raw_file):
        logger.warning(f"⚠ Raw data file missing for {ticker}. Skipping...")
        return None

    df = pd.read_csv(raw_file)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df.set_index("Date", inplace=True)
    else:
        logger.warning(f"⚠ No valid 'Date' column in {ticker}.csv. Skipping...")
        return None

    df.columns = df.columns.str.lower()
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        logger.warning(f"⚠ Missing columns {missing_cols} in {ticker}.csv. Skipping...")
        return None

    return df


@handle_exceptions
def compute_technical_indicators(ticker):
    """Computes technical indicators for a given stock ticker."""
    df = read_stock_data(ticker)
    if df is None:
        return

    indicators = {
        "SMA_50": ta.sma(df["close"], length=50),
        "SMA_200": ta.sma(df["close"], length=200),
        "EMA_50": ta.ema(df["close"], length=50),
        "EMA_200": ta.ema(df["close"], length=200),
        "RSI_14": ta.rsi(df["close"], length=14),
        "ADX_14": ta.adx(df["high"], df["low"], df["close"], length=14),
        "ATR_14": ta.atr(df["high"], df["low"], df["close"], length=14),
        "CCI_20": ta.cci(df["high"], df["low"], df["close"], length=20),
        "WilliamsR_14": ta.willr(df["high"], df["low"], df["close"], length=14),
        "MACD": ta.macd(df["close"]),
        "Bollinger": ta.bbands(df["close"], length=20),
        "Stochastic": ta.stoch(df["high"], df["low"], df["close"], k=14, d=3),
    }

    # ✅ Ensure indicators exist before assignment
    df = df.assign(
        MACD=(
            indicators["MACD"]["MACD_12_26_9"]
            if indicators["MACD"] is not None
            else None
        ),
        MACD_Signal=(
            indicators["MACD"]["MACDs_12_26_9"]
            if indicators["MACD"] is not None
            else None
        ),
        MACD_Hist=(
            indicators["MACD"]["MACDh_12_26_9"]
            if indicators["MACD"] is not None
            else None
        ),
        BB_Upper=(
            indicators["Bollinger"]["BBU_20_2.0"]
            if indicators["Bollinger"] is not None
            else None
        ),
        BB_Middle=(
            indicators["Bollinger"]["BBM_20_2.0"]
            if indicators["Bollinger"] is not None
            else None
        ),
        BB_Lower=(
            indicators["Bollinger"]["BBL_20_2.0"]
            if indicators["Bollinger"] is not None
            else None
        ),
        Stoch_K=(
            indicators["Stochastic"]["STOCHk_14_3_3"]
            if indicators["Stochastic"] is not None
            else None
        ),
        Stoch_D=(
            indicators["Stochastic"]["STOCHd_14_3_3"]
            if indicators["Stochastic"] is not None
            else None
        ),
    )

    df.dropna(inplace=True)
    processed_file = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
    df.to_csv(processed_file)
    logger.info(f"✅ Processed indicators for {ticker} saved to {processed_file}")


@handle_exceptions
def process_all_stocks():
    """Computes technical indicators for all available stock data."""
    tickers = [
        file.replace(".csv", "")
        for file in os.listdir(RAW_DATA_DIR)
        if file.endswith(".csv")
    ]
    console.print(
        Panel(
            f"🚀 Computing technical indicators for {len(tickers)} stocks...",
            style="bold cyan",
        )
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
                executor.submit(compute_technical_indicators, ticker): ticker
                for ticker in tickers
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                progress.update(task, advance=1)

    console.print(Panel("✅ Done computing technical indicators!", style="bold green"))


if __name__ == "__main__":
    process_all_stocks()
