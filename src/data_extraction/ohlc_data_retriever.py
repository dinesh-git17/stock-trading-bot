import concurrent.futures
import logging
import os
import time

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

# ✅ Import utilities
from src.tools.utils import handle_exceptions, setup_logging

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/ohlc_data.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)
console = Console()
logger.info("🚀 Logging setup complete.")

# ✅ Constants
DATA_DIR = "data/raw/"
os.makedirs(DATA_DIR, exist_ok=True)

MAX_RETRIES = 5
THREADS = 4
RETRY_DELAY = 2
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume", "adjusted_close"]

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


@handle_exceptions
def clean_ticker(ticker):
    """Ensure tickers are stored as clean strings (fix tuple issue)."""
    return ticker[0] if isinstance(ticker, tuple) else ticker


@handle_exceptions
def ensure_all_columns(df):
    """Ensure all required columns are present by adding missing ones as NaN."""
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[REQUIRED_COLUMNS]


@handle_exceptions
def save_stock_data(ticker, df):
    """Save stock data to CSV, ensuring date formatting and column consistency."""
    if df is None or df.empty:
        logger.warning(f"⚠ No valid data to save for {ticker}. Skipping...")
        return False

    df = df.sort_index()
    df = ensure_all_columns(df)
    df.to_csv(f"{DATA_DIR}/{ticker}.csv", index=True)
    logger.info(f"✅ Data saved for {ticker}")
    return True


@handle_exceptions
def fetch_stock_data_alpha_vantage(ticker):
    """Fetch stock data from Alpha Vantage."""
    ticker = clean_ticker(ticker)

    if not ALPHA_VANTAGE_API_KEY:
        logger.error("⚠ Alpha Vantage API key is missing! Check .env file.")
        return None

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "compact",
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if "Time Series (Daily)" not in data:
                logger.warning(f"⚠ No data found for {ticker} on Alpha Vantage.")
                return None

            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.index = pd.to_datetime(df.index)
            df.rename(
                columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. adjusted close": "adjusted_close",
                    "6. volume": "volume",
                },
                inplace=True,
            )
            return df
        except requests.exceptions.RequestException as e:
            logger.error(
                f"⚠ Alpha Vantage error fetching {ticker} (Attempt {attempt+1}): {e}"
            )
            time.sleep(RETRY_DELAY * (attempt + 1))
    return None


@handle_exceptions
def fetch_stock_data_yahoo(ticker, start="2015-01-01", end=None):
    """Fetch stock data from Yahoo Finance."""
    ticker = clean_ticker(ticker)
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=True)
        if df.empty:
            logger.warning(f"⚠ No data found for {ticker} on Yahoo Finance.")
            return None

        df.index = pd.to_datetime(df.index)
        df["adjusted_close"] = df["Close"]
        df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )
        return df[REQUIRED_COLUMNS]
    except Exception as e:
        logger.error(f"⚠ Yahoo Finance error fetching {ticker}: {e}")
        return None


@handle_exceptions
def fetch_stock_data(ticker):
    """Try multiple sources to fetch stock data."""
    ticker = clean_ticker(ticker)
    df = fetch_stock_data_alpha_vantage(ticker)
    if df is not None and save_stock_data(ticker, df):
        return ticker
    df = fetch_stock_data_yahoo(ticker)
    if df is not None and save_stock_data(ticker, df):
        return ticker
    return None


@handle_exceptions
def fetch_ohlc_data(tickers):
    """Fetch OHLC data concurrently for multiple tickers."""
    console.print(Panel("🚀 Fetching OHLCV data...", style="bold yellow"))
    successful_tickers, failed_tickers = [], tickers[:]

    retry_attempt = 0
    while failed_tickers and retry_attempt < MAX_RETRIES:
        retry_attempt += 1
        console.print(
            Panel(
                f"🔄 Retry {retry_attempt}: Fetching {len(failed_tickers)} failed tickers...",
                style="bold red",
            )
        )

        with Progress(
            TextColumn("[bold yellow]Fetching OHLCV:[/bold yellow]"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching", total=len(failed_tickers))
            with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
                futures = {
                    executor.submit(fetch_stock_data, ticker): ticker
                    for ticker in failed_tickers
                }
                new_failed_tickers = [
                    futures[future]
                    for future in concurrent.futures.as_completed(futures)
                    if future.result() is None
                ]
                successful_tickers.extend(set(failed_tickers) - set(new_failed_tickers))
                progress.update(
                    task, advance=len(failed_tickers) - len(new_failed_tickers)
                )
        failed_tickers = list(set(new_failed_tickers))
        time.sleep(RETRY_DELAY * retry_attempt)
    console.print(
        Panel(
            f"✅ Successfully fetched data for {len(successful_tickers)} tickers!",
            style="bold green",
        )
    )


if __name__ == "__main__":
    from src.data_extraction.stock_data_collector import fetch_most_active_stocks

    most_active_stocks = fetch_most_active_stocks()
    fetch_ohlc_data(most_active_stocks)
