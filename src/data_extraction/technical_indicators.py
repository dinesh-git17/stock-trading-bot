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

    # ✅ Compute Indicators
    df = df.assign(
        sma_50=ta.sma(df["close"], length=50),
        sma_200=ta.sma(df["close"], length=200),
        ema_50=ta.ema(df["close"], length=50),
        ema_200=ta.ema(df["close"], length=200),
        rsi_14=ta.rsi(df["close"], length=14),
        adx_14=ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"],
        atr_14=ta.atr(df["high"], df["low"], df["close"], length=14),
        cci_20=ta.cci(df["high"], df["low"], df["close"], length=20),
        williamsr_14=ta.willr(df["high"], df["low"], df["close"], length=14),
    )

    # ✅ Handle MACD, Bollinger Bands, and Stochastic
    macd = ta.macd(df["close"])
    bbands = ta.bbands(df["close"], length=20)
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)

    df = df.assign(
        macd=macd["MACD_12_26_9"] if macd is not None else None,
        macd_signal=macd["MACDs_12_26_9"] if macd is not None else None,
        macd_hist=macd["MACDh_12_26_9"] if macd is not None else None,
        bb_upper=bbands["BBU_20_2.0"] if bbands is not None else None,
        bb_middle=bbands["BBM_20_2.0"] if bbands is not None else None,
        bb_lower=bbands["BBL_20_2.0"] if bbands is not None else None,
        stoch_k=stoch["STOCHk_14_3_3"] if stoch is not None else None,
        stoch_d=stoch["STOCHd_14_3_3"] if stoch is not None else None,
    )

    # ✅ Drop NaN values after indicator calculations
    df.dropna(inplace=True)

    # ✅ Save processed data
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
