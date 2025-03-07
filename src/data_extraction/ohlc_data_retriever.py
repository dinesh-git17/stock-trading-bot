import yfinance as yf
import pandas as pd
import os
import logging
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup logging
logging.basicConfig(
    filename="data/logs/ohlc_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

DATA_DIR = "data/raw/"
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure directory exists

def fetch_ohlc_data(tickers, start="2023-01-01", end=None):
    """
    Fetch OHLCV data for a list of stock tickers.
    Saves each stock's data as a CSV file inside data/raw/.
    """
    console.print("\n")
    with Progress(SpinnerColumn(), TextColumn("[bold yellow]Fetching OHLCV data...[/bold yellow]"), console=console) as progress:
        task = progress.add_task("fetching", total=len(tickers))

        for ticker_tuple in tickers:
            try:
                ticker = ticker_tuple[0]  # ✅ Extract only the ticker from tuple

                stock = yf.Ticker(ticker)
                df = stock.history(start=start, end=end, auto_adjust=True)

                if df.empty:
                    logging.warning(f"No data found for {ticker}")
                    continue  # Skip if no data

                file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
                df.to_csv(file_path)

                logging.info(f"Saved {ticker} OHLCV data to {file_path}")

            except Exception as e:
                logging.error(f"Failed to fetch {ticker}: {e}")

            progress.update(task, advance=1)
            time.sleep(1)  # Avoid hitting API rate limits

    console.print("[bold green]✅ Done fetching OHLCV data![/bold green]\n")

if __name__ == "__main__":
    # Load most active stocks from previous step
    from stock_data_collector import fetch_most_active_stocks
    most_active_stocks = fetch_most_active_stocks()

    fetch_ohlc_data(most_active_stocks)
