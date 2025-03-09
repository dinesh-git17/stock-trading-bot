import concurrent.futures
import logging
import os
import shutil
import time

import yfinance as yf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Setup logging
os.makedirs("data/logs", exist_ok=True)
logging.basicConfig(
    filename="data/logs/ohlc_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()
DATA_DIR = "data/raw/"
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure directory exists


def fetch_single_ohlc(ticker, start="2023-01-01", end=None, retries=3):
    """
    Fetch OHLCV data for a single stock ticker with retry logic.
    Saves data as CSV in 'data/raw/'.
    """
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")

    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=end, auto_adjust=True)

            if df.empty:
                logging.warning(f"No data found for {ticker}")
                return None  # Skip invalid ticker

            df.to_csv(file_path)
            logging.info(f"Saved {ticker} OHLCV data to {file_path}")
            return ticker  # Success

        except Exception as e:
            logging.error(f"Attempt {attempt+1}/{retries} failed for {ticker}: {e}")
            time.sleep(2)  # Small delay before retrying

    return None  # Failed after retries


def fetch_ohlc_data(tickers, start="2023-01-01", end=None):
    """
    Fetch OHLCV data for a list of stock tickers in parallel.
    Uses threading for faster execution.
    """
    console.print("\n")
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Fetching OHLCV data...[/bold yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching OHLCV", total=len(tickers))

        valid_tickers = [t[0] for t in tickers]  # Extract ticker symbols from tuples

        # Use ThreadPoolExecutor to speed up requests
        successful_tickers = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                executor.map(lambda t: fetch_single_ohlc(t, start, end), valid_tickers)
            )

        # Collect successfully fetched tickers
        successful_tickers = [ticker for ticker in results if ticker is not None]

        progress.update(task, advance=len(valid_tickers), completed=len(valid_tickers))

    console.print(
        f"[bold green]✅ Fetched OHLCV data for {len(successful_tickers)} tickers![/bold green]\n"
    )
    display_summary(successful_tickers)


def display_summary(successful_tickers):
    """
    Display summary table of successfully fetched stock data.
    Dynamically adjusts column count based on terminal size.
    """
    if not successful_tickers:
        console.print("[bold red]❌ No stock data was retrieved.[/bold red]")
        return

    # Determine terminal width & optimal column count
    terminal_width = shutil.get_terminal_size((100, 20)).columns
    max_columns = max(2, terminal_width // 20)  # ~20 chars per column
    rows_per_table = 10  # Show fewer rows per table

    console.print(
        f"[bold cyan]📊 Displaying fetched stock tickers in {max_columns} columns.[/bold cyan]\n"
    )

    # Split tickers into table rows
    table = Table(title="Successfully Fetched OHLCV Data", show_lines=True)

    # Create column headers dynamically
    for i in range(max_columns):
        table.add_column(f"Ticker {i+1}", justify="center", style="bold cyan")

    # Fill table rows dynamically
    for i in range(0, len(successful_tickers), max_columns):
        row_tickers = successful_tickers[i : i + max_columns]
        while len(row_tickers) < max_columns:
            row_tickers.append("")  # Fill empty spaces to align columns
        table.add_row(*row_tickers)

        # Display the table every `rows_per_table` rows for better readability
        if (i // max_columns) % rows_per_table == 0 and i > 0:
            console.print(table)
            table = Table(
                title="Successfully Fetched OHLCV Data (Continued)", show_lines=True
            )
            for i in range(max_columns):
                table.add_column(f"Ticker {i+1}", justify="center", style="bold cyan")

    console.print(table)
    console.print("\n")


if __name__ == "__main__":
    # Load most active stocks from previous step
    from stock_data_collector import fetch_most_active_stocks

    most_active_stocks = fetch_most_active_stocks()
    fetch_ohlc_data(most_active_stocks)
    fetch_ohlc_data(most_active_stocks)
