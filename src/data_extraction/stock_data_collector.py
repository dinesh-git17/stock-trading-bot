import concurrent.futures
import logging
from io import StringIO
from time import sleep

import pandas as pd
import requests
import yfinance as yf
from rich.columns import Columns
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Setup logging
logging.basicConfig(
    filename="data/logs/stock_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()
YAHOO_MOST_ACTIVE_URL = "https://finance.yahoo.com/markets/stocks/most-active"


def fetch_most_active_stocks(limit=100):
    """
    Fetch at least 'limit' most active stocks from Yahoo Finance.
    Uses pagination to retrieve data efficiently.
    Returns a list of valid stock tickers along with company names.
    """
    all_stocks = set()  # Use a set to avoid duplicates
    start = 0  # Pagination start

    console.print("\n")
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Fetching most active stocks...[/bold yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching stocks", total=None)

        try:
            while len(all_stocks) < limit:
                url = f"{YAHOO_MOST_ACTIVE_URL}/?start={start}&count=25"
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()

                # Read the table efficiently using lxml
                tables = pd.read_html(StringIO(response.text), flavor="lxml")

                if not tables:
                    console.print(
                        "\n[bold red]No more data available from Yahoo![/bold red]\n"
                    )
                    break

                data = tables[0][["Symbol", "Name"]]
                stocks = {
                    (row["Symbol"].upper(), row["Name"]) for _, row in data.iterrows()
                }

                all_stocks.update(stocks)
                start += 25  # Move to next batch

                # Stop if fewer than 25 new stocks were found (end of data)
                if len(stocks) < 25:
                    break

            # Convert set to a list and keep only 'limit' stocks
            all_stocks = list(all_stocks)[:limit]

            # Validate tickers in parallel
            valid_stocks = validate_stocks_parallel(all_stocks)

            progress.update(task, advance=1, completed=1, visible=False)
            console.print(
                f"[bold green]✅ {len(valid_stocks)} valid stocks fetched![/bold green]\n"
            )

            logging.info(f"Fetched {len(valid_stocks)} most active stocks.")
            return valid_stocks

        except Exception as e:
            console.print(f"\n[bold red]Error fetching stocks:[/bold red] {e}\n")
            logging.error(f"Error fetching stocks: {e}")
            return []


def validate_stock(ticker, name):
    """
    Validate stock tickers by checking if Yahoo Finance has valid market data.
    Uses a retry mechanism for reliability.
    Returns the ticker and name if valid, otherwise None.
    """
    stock = yf.Ticker(ticker)
    retries = 3
    for attempt in range(retries):
        try:
            # Check multiple possible sources for price data
            last_price = stock.fast_info.get("last_price")
            if last_price is None:
                last_price = stock.history(period="1d").get("Close")
                if last_price is not None and not last_price.empty:
                    last_price = last_price.iloc[-1]

            if last_price is not None:
                return ticker, name
        except Exception as e:
            logging.warning(f"Retry {attempt + 1}/{retries} for {ticker}: {e}")
            sleep(2)  # Wait before retrying

    return None  # If all retries fail, consider invalid


def validate_stocks_parallel(stocks):
    """
    Validate stock tickers in parallel using ThreadPoolExecutor.
    """
    valid_stocks = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda s: validate_stock(*s), stocks)

    # Filter out None values
    valid_stocks = [res for res in results if res is not None]

    return valid_stocks


def display_stocks(stocks):
    """
    Display fetched stock tickers & company names using dynamically formatted tables.
    """

    stocks_per_table = 25
    tables = []

    for i in range(0, len(stocks), stocks_per_table):
        table = Table(
            title=f"Stocks {i+1}-{min(i+stocks_per_table, len(stocks))}",
            show_lines=True,
        )
        table.add_column("Stock Symbol", justify="center", style="cyan")
        table.add_column("Company Name", justify="center", style="magenta")

        for ticker, name in stocks[i : i + stocks_per_table]:
            table.add_row(ticker, name)

        tables.append(table)

    console.print("\n")
    console.print(Columns(tables, expand=True))
    console.print("\n")


if __name__ == "__main__":
    most_active_stocks = fetch_most_active_stocks()
    if most_active_stocks:
        display_stocks(most_active_stocks)
