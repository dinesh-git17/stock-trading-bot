import logging
import os
import signal
import sys
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from rich.console import Console

from src.data_extraction.ohlc_data_retriever import fetch_ohlc_data
from src.data_extraction.stock_data_collector import fetch_most_active_stocks
from src.data_extraction.technical_indicators import process_all_stocks

# ✅ Import utilities
from src.tools.utils import handle_exceptions, setup_logging

# ✅ Setup Logging
LOG_FILE = "data/logs/scheduler.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)
console = Console()
logger.info("🚀 Logging setup complete.")

# ✅ Initialize Scheduler
scheduler = BackgroundScheduler()


@handle_exceptions
def update_stock_data():
    """Scheduled task to update stock data every hour."""
    console.print("[bold yellow]🔄 Updating stock data...[/bold yellow]")
    logger.info("Updating stock data...")

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
        logger.info("Stock data update completed successfully.")
    except Exception as e:
        console.print(f"[bold red]❌ Error updating stock data:[/bold red] {e}")
        logger.error(f"Error updating stock data: {e}", exc_info=True)


@handle_exceptions
def start_scheduler():
    """Starts the APScheduler to run the update_stock_data function every hour."""
    console.print(
        "\n[bold cyan]📅 Starting automated stock data updates (every hour)...[/bold cyan]\n"
    )
    logger.info("Scheduler started. Running stock updates every hour.")

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
        logger.info("Scheduler shutting down...")
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
