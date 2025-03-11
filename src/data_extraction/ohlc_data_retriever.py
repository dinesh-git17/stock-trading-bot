import concurrent.futures
import logging
import os
import time

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/ohlc_data.log"
os.makedirs("data/logs", exist_ok=True)

# ✅ Insert 5 blank lines before logging new logs
with open(LOG_FILE, "a") as log_file:
    log_file.write("\n" * 5)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()
DATA_DIR = "data/raw/"
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure directory exists

# ✅ Use environment variables for API keys (SAFER)
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# ✅ Constants
MAX_RETRIES = 5  # 🔥 Retries for failed requests
THREADS = 4  # ✅ Limit to prevent connection pool exhaustion
RETRY_DELAY = 2  # ✅ Start with 2s delay for retries
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume", "adjusted_close"]

### **🚀 Stock Data Fetching Methods**


def clean_ticker(ticker):
    """Ensure tickers are stored as clean strings (fix tuple issue)."""
    return ticker[0] if isinstance(ticker, tuple) else ticker


def ensure_all_columns(df):
    """Ensure all required columns are present by adding missing ones as NaN."""
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None  # Add missing column with NaN
    return df[REQUIRED_COLUMNS]  # Reorder for consistency


def save_stock_data(ticker, df):
    """
    Saves stock data to CSV, ensuring the date column is properly formatted.
    """
    if df is None or df.empty:
        logging.warning(f"⚠ No valid data to save for {ticker}. Skipping...")
        return False

    df = df.sort_index()  # ✅ Ensure dates are sorted
    df = ensure_all_columns(df)  # ✅ Ensure all columns are present

    df.to_csv(f"{DATA_DIR}/{ticker}.csv", index=True)  # ✅ Keep date index in CSV
    logging.info(f"✅ Data saved for {ticker}")
    return True


def fetch_stock_data_alpha_vantage(ticker):
    """
    Fetch historical stock data from Alpha Vantage.
    """
    ticker = clean_ticker(ticker)

    if not ALPHA_VANTAGE_API_KEY:
        logging.error("⚠ Alpha Vantage API key is missing! Check .env file.")
        return None

    url = f"https://www.alphavantage.co/query"
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
                logging.warning(f"⚠ No data found for {ticker} on Alpha Vantage.")
                return None

            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.index = pd.to_datetime(df.index)  # ✅ Set proper date index
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
            )  # ✅ Standardize column names
            return df

        except requests.exceptions.RequestException as e:
            logging.error(
                f"⚠ Alpha Vantage error fetching {ticker} (Attempt {attempt+1}): {e}"
            )
            time.sleep(RETRY_DELAY * (attempt + 1))  # ✅ Exponential backoff

    return None


def fetch_stock_data_yahoo(ticker, start="2015-01-01", end=None):
    """
    Fetch stock data from Yahoo Finance using yfinance.
    """
    ticker = clean_ticker(ticker)

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=True)

        if df.empty:
            logging.warning(f"⚠ No data found for {ticker} on Yahoo Finance.")
            return None

        df.index = pd.to_datetime(df.index)  # ✅ Ensure dates are proper
        df["adjusted_close"] = df["Close"]  # ✅ Ensure adjusted close is included
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
        return df[REQUIRED_COLUMNS]  # ✅ Ensure correct column order

    except Exception as e:
        logging.error(f"⚠ Yahoo Finance error fetching {ticker}: {e}")
        return None


def fetch_stock_data(ticker):
    """
    Tries multiple methods to fetch stock data.
    Order: Alpha Vantage -> Yahoo Finance
    """
    ticker = clean_ticker(ticker)

    df = fetch_stock_data_alpha_vantage(ticker)
    if df is not None and save_stock_data(ticker, df):
        return ticker

    df = fetch_stock_data_yahoo(ticker)
    if df is not None and save_stock_data(ticker, df):
        return ticker

    return None


### **🚀 Multi-Threaded Data Fetching**
def fetch_ohlc_data(tickers):
    console.print("\n[bold yellow]🚀 Fetching OHLCV data...[/bold yellow]\n")

    successful_tickers = []
    failed_tickers = tickers[:]

    retry_attempt = 0
    while failed_tickers and retry_attempt < MAX_RETRIES:
        retry_attempt += 1
        console.print(
            f"\n[bold yellow]🔄 Retry {retry_attempt}: Fetching {len(failed_tickers)} failed tickers...[/bold yellow]"
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

                new_failed_tickers = []
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        successful_tickers.append(result)
                    else:
                        new_failed_tickers.append(futures[future])

                    progress.update(task, advance=1)

        failed_tickers = list(set(new_failed_tickers))  # Remove duplicates
        time.sleep(RETRY_DELAY * retry_attempt)

    console.print(
        f"\n[bold green]✅ Successfully fetched data for {len(successful_tickers)} tickers![/bold green]\n"
    )


### **🚀 Run the Script**
if __name__ == "__main__":
    from stock_data_collector import fetch_most_active_stocks

    most_active_stocks = fetch_most_active_stocks()
    fetch_ohlc_data(most_active_stocks)
