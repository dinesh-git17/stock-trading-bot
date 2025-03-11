import logging
import os
import random
import shutil
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table

# ✅ Setup Logging
os.makedirs("data/logs", exist_ok=True)
logging.basicConfig(
    filename="data/logs/stock_data_collector.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ Constants
MOST_ACTIVE_LIMIT = 100
EXPANDED_LIMIT = 150
YAHOO_FINANCE_MOST_ACTIVE_URL = "https://finance.yahoo.com/most-active"
YAHOO_FINANCE_GAINERS_URL = "https://finance.yahoo.com/gainers"
YAHOO_FINANCE_TRENDING_URL = "https://finance.yahoo.com/trending-tickers"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
}


def validate_ticker(ticker):
    """Checks if a stock ticker is valid by attempting to fetch its 1-day history."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1d")
        return not df.empty  # ✅ Return True if data exists, False otherwise
    except Exception:
        return False


def fetch_stock_list(url, source_name, retries=3):
    """Fetches stock tickers from Yahoo Finance's most active, gainers, or trending lists."""
    for attempt in range(retries):
        try:
            req = Request(url, headers=HEADERS)
            response = urlopen(req)
            tables = pd.read_html(response.read())

            if tables and not tables[0].empty:
                console.print(
                    f"[bold green]✅ Fetched {source_name} stocks![/bold green]"
                )
                return tables[0]["Symbol"].tolist()
            else:
                return []

        except HTTPError as e:
            if e.code == 429:
                console.print(
                    f"[bold red]❌ Rate limited! Retrying {source_name} in {2**attempt} seconds...[/bold red]"
                )
                time.sleep(2**attempt)  # ✅ Exponential backoff (2s, 4s, 8s)
            else:
                logging.error(f"⚠ HTTP Error ({source_name}): {e}")
                return []

        except Exception as e:
            logging.error(f"⚠ Error fetching {source_name}: {e}")
            return []

        time.sleep(random.uniform(1, 3))  # ✅ Random delay to prevent detection

    return []


def fetch_most_active_stocks():
    """Fetches the most active, trending, and gainers stocks from Yahoo Finance."""
    console.print("[bold yellow]📊 Fetching the most active stocks...[/bold yellow]")

    # ✅ Fetch stock tickers from multiple sources
    most_active = fetch_stock_list(YAHOO_FINANCE_MOST_ACTIVE_URL, "Most Active")
    gainers = fetch_stock_list(YAHOO_FINANCE_GAINERS_URL, "Top Gainers")
    trending = fetch_stock_list(YAHOO_FINANCE_TRENDING_URL, "Trending Tickers")

    # ✅ Combine stock lists and remove duplicates
    all_stocks = list(
        dict.fromkeys(most_active + gainers + trending)
    )  # ✅ Keep order, remove duplicates

    console.print(
        f"[bold cyan]🔍 Found {len(all_stocks)} unique stock tickers before validation.[/bold cyan]"
    )

    # ✅ Validate tickers (ensure they have trade data)
    valid_stocks = [
        ticker for ticker in all_stocks[:EXPANDED_LIMIT] if validate_ticker(ticker)
    ]

    # ✅ Ensure we get at least 100 valid stocks
    final_stocks = valid_stocks[:MOST_ACTIVE_LIMIT]

    console.print(
        f"[bold green]✅ {len(final_stocks)} valid stocks fetched![/bold green]"
    )
    logging.info(f"✅ {len(final_stocks)} valid stocks fetched.")

    display_active_stocks(final_stocks)
    return [(ticker,) for ticker in final_stocks]


def display_active_stocks(valid_stocks):
    """Displays the most active stocks in a structured table based on terminal width."""
    terminal_width = shutil.get_terminal_size((100, 20)).columns
    max_columns = max(
        3, terminal_width // 12
    )  # ✅ Adjust columns based on terminal width

    table = Table(title="Most Active Stocks", show_lines=False)
    for i in range(max_columns):
        table.add_column(f"Ticker {i+1}", justify="center", style="bold cyan")

    # ✅ Split tickers into table rows
    for i in range(0, len(valid_stocks), max_columns):
        row_tickers = valid_stocks[i : i + max_columns]
        while len(row_tickers) < max_columns:
            row_tickers.append("")  # ✅ Fill empty spaces to align columns
        table.add_row(*row_tickers)

    console.print(table)


if __name__ == "__main__":
    most_active_stocks = fetch_most_active_stocks()
    console.print(
        f"[bold cyan]🎯 Collected {len(most_active_stocks)} most active stocks![/bold cyan]"
    )
