import logging
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ✅ Import utilities
from src.tools.utils import handle_exceptions, setup_logging

# ✅ Setup Logging
LOG_FILE = "data/logs/stock_data_collector.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)  # Ensure log directory exists
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)
console = Console()
logger.info("🚀 Logging setup complete.")

# ✅ Constants
MOST_ACTIVE_LIMIT = 100
EXPANDED_LIMIT = 150
YAHOO_FINANCE_SOURCES = {
    "Most Active": "https://finance.yahoo.com/most-active",
    "Top Gainers": "https://finance.yahoo.com/gainers",
    "Trending Tickers": "https://finance.yahoo.com/trending-tickers",
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
}


@handle_exceptions
def validate_ticker(ticker):
    """Checks if a stock ticker is valid by attempting to fetch its 1-day history."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1d")
        return ticker if not df.empty else None  # ✅ Return ticker if valid, else None
    except Exception as e:
        logger.error(f"⚠ Error validating ticker {ticker}: {e}")
        return None


@handle_exceptions
def fetch_stock_list(url, source_name, retries=3):
    """Fetches stock tickers from Yahoo Finance sources with retries."""
    for attempt in range(retries):
        try:
            req = Request(url, headers=HEADERS)
            response = urlopen(req)
            tables = pd.read_html(response.read())

            if tables and not tables[0].empty:
                console.print(
                    Panel(
                        f"✅ Successfully fetched {source_name} stocks!",
                        style="bold green",
                    )
                )
                return set(tables[0]["Symbol"].astype(str).tolist())
            else:
                return set()

        except HTTPError as e:
            if e.code == 429:
                console.print(
                    Panel(
                        f"❌ Rate limited! Retrying {source_name} in {2**attempt} seconds...",
                        style="bold red",
                    )
                )
                time.sleep(2**attempt)  # ✅ Exponential backoff (2s, 4s, 8s)
            else:
                logger.error(f"⚠ HTTP Error ({source_name}): {e}")
                return set()

        except Exception as e:
            logger.error(f"⚠ Error fetching {source_name}: {e}")
            return set()

        time.sleep(random.uniform(1, 3))  # ✅ Random delay to prevent detection
    return set()


@handle_exceptions
def fetch_most_active_stocks():
    """Fetches and validates the most active stocks from multiple sources."""
    console.print(Panel("📊 Fetching the most active stocks...", style="bold yellow"))

    # ✅ Fetch stock tickers from all sources concurrently
    with ThreadPoolExecutor() as executor:
        stock_results = executor.map(
            lambda source: fetch_stock_list(YAHOO_FINANCE_SOURCES[source], source),
            YAHOO_FINANCE_SOURCES.keys(),
        )

    all_stocks = set().union(*stock_results)  # ✅ Merge all sets (unique tickers only)
    console.print(
        Panel(
            f"🔍 Found {len(all_stocks)} unique stock tickers before validation.",
            style="bold cyan",
        )
    )

    # ✅ Validate tickers concurrently
    with ThreadPoolExecutor() as executor:
        valid_stocks = sorted(
            filter(
                None, executor.map(validate_ticker, list(all_stocks)[:EXPANDED_LIMIT])
            )
        )

    # ✅ Ensure at least 100 valid stocks
    final_stocks = valid_stocks[:MOST_ACTIVE_LIMIT]

    console.print(
        Panel(f"✅ {len(final_stocks)} valid stocks fetched!", style="bold green")
    )
    logger.info(
        f"✅ {len(final_stocks)} valid stocks fetched: {', '.join(final_stocks)}"
    )

    display_active_stocks(final_stocks)
    return [(ticker,) for ticker in final_stocks]


@handle_exceptions
def display_active_stocks(valid_stocks):
    """Displays the most active stocks in a structured table based on terminal width."""
    terminal_width = shutil.get_terminal_size((100, 20)).columns
    max_columns = max(
        3, terminal_width // 12
    )  # ✅ Adjust columns based on terminal width

    table = Table(title="Most Active Stocks", show_lines=False)
    for i in range(max_columns):
        table.add_column(f"Ticker {i+1}", justify="center", style="bold cyan")

    # ✅ Sort tickers before displaying
    sorted_stocks = sorted(valid_stocks)

    # ✅ Split tickers into table rows
    for i in range(0, len(sorted_stocks), max_columns):
        row_tickers = [str(ticker) for ticker in sorted_stocks[i : i + max_columns]]
        while len(row_tickers) < max_columns:
            row_tickers.append("")  # ✅ Fill empty spaces to align columns
        table.add_row(*row_tickers)

    console.print(table)


if __name__ == "__main__":
    most_active_stocks = fetch_most_active_stocks()
    console.print(
        Panel(
            f"🎯 Collected {len(most_active_stocks)} most active stocks!",
            style="bold cyan",
        )
    )
