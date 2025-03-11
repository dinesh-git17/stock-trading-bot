import logging
import signal
import sys
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from ohlc_data_retriever import fetch_ohlc_data
from rich.console import Console
from stock_data_collector import fetch_most_active_stocks
from technical_indicators import process_all_stocks

# ✅ Setup Logging
LOG_FILE = "data/logs/scheduler.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()
scheduler = BackgroundScheduler()


def update_stock_data():
    """
    Scheduled task to update stock data every hour.
    Fetches most active stocks, retrieves OHLCV data, and updates technical indicators.
    """
    console.print("[bold yellow]🔄 Updating stock data...[/bold yellow]")
    logging.info("Updating stock data...")

    try:
        # ✅ Fetch new trending stocks
        most_active_stocks = fetch_most_active_stocks()

        if not most_active_stocks:
            raise ValueError("No active stocks retrieved.")

        # ✅ Update OHLCV data
        fetch_ohlc_data(most_active_stocks)

        # ✅ Recalculate technical indicators
        process_all_stocks()

        console.print("[bold green]✅ Stock data update complete![/bold green]")
        logging.info("Stock data update completed successfully.")

    except Exception as e:
        console.print(f"[bold red]❌ Error updating stock data:[/bold red] {e}")
        logging.error(f"Error updating stock data: {e}", exc_info=True)


def start_scheduler():
    """
    Starts the APScheduler to run the update_stock_data function every hour.
    """
    console.print(
        "\n[bold cyan]📅 Starting automated stock data updates (every hour)...[/bold cyan]\n"
    )
    logging.info("Scheduler started. Running stock updates every hour.")

    # ✅ Schedule job with error handling
    scheduler.add_job(
        update_stock_data,
        IntervalTrigger(hours=1),
        id="stock_update",
        replace_existing=True,
    )

    scheduler.start()

    # ✅ Graceful exit handling
    def graceful_shutdown(signum, frame):
        console.print("\n[bold red]⏹ Scheduler stopping...[/bold red]")
        logging.info("Scheduler shutting down...")
        scheduler.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_shutdown)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, graceful_shutdown)  # Handle termination signals

    try:
        while True:
            time.sleep(1)  # Keep script running
    except (KeyboardInterrupt, SystemExit):
        graceful_shutdown(None, None)


if __name__ == "__main__":
    start_scheduler()
