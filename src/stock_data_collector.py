import yfinance as yf
import pandas as pd
import requests
import logging
import shutil
from io import StringIO
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.columns import Columns

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
    Fetch at least 'limit' most active stocks from Yahoo Finance by iterating pages.
    Returns a list of valid stock tickers along with company names.
    """
    all_stocks = []
    start = 0  # Pagination starts at 0

    console.print("\n")  # Add vertical space
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Fetching most active stocks...[/bold yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("fetching", total=None)  # Indeterminate progress bar

        try:
            while len(all_stocks) < limit:
                url = f"{YAHOO_MOST_ACTIVE_URL}/?start={start}&count=25"
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()

                # Read the table using StringIO
                tables = pd.read_html(StringIO(response.text))

                if len(tables) == 0:
                    console.print(
                        "\n[bold red]No more data available from Yahoo![/bold red]\n"
                    )
                    break  # No more data to fetch

                data = tables[0][["Symbol", "Name"]]
                stocks = [
                    (row["Symbol"].upper(), row["Name"]) for _, row in data.iterrows()
                ]

                all_stocks.extend(stocks)
                start += 25  # Move to the next page

                # Stop if Yahoo returns fewer than 25 stocks (end of data)
                if len(stocks) < 25:
                    break

            # Ensure we have exactly 'limit' stocks
            all_stocks = list(dict.fromkeys(all_stocks))[
                :limit
            ]  # Remove duplicates, keep order

            # Validate tickers
            valid_stocks = validate_stocks(all_stocks)

            # Mark progress as done
            progress.update(task, advance=1, completed=1, visible=False)
            console.print(
                "[bold green]✅ Done fetching most active stocks![/bold green]\n"
            )

            # Log successful fetch
            logging.info(f"Fetched {len(valid_stocks)} most active stocks.")

            return valid_stocks

        except Exception as e:
            console.print(f"\n[bold red]Error fetching stocks:[/bold red] {e}\n")
            logging.error(f"Error fetching stocks: {e}")
            return []


def validate_stocks(stocks):
    """
    Validate stock tickers by checking if data exists in Yahoo Finance.
    """
    valid_stocks = []
    for ticker, name in stocks:
        stock = yf.Ticker(ticker)
        try:
            if "regularMarketPrice" in stock.info:
                valid_stocks.append((ticker, name))
        except:
            pass  # Ignore invalid tickers

    return valid_stocks


def display_stocks(stocks):
    """
    Display fetched stock tickers & company names in multiple Rich tables side by side.
    """
    terminal_size = shutil.get_terminal_size((100, 20))  # Default to (100, 20) if None
    terminal_width = (
        terminal_size.columns if terminal_size else 100
    )  # Fix for NoneType error

    num_tables = max(2, terminal_width // 40)  # Determine number of side-by-side tables
    stocks_per_table = 25  # Rows per table
    tables = []

    # Create tables with 25 stocks each
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

    # Display tables side by side
    console.print("\n")
    console.print(Columns(tables, expand=True))
    console.print("\n")


if __name__ == "__main__":
    most_active_stocks = fetch_most_active_stocks()
    display_stocks(most_active_stocks)  # ✅ Fix: Removed () after most_active_stocks
