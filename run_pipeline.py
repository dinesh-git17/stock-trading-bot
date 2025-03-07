import os
import subprocess
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup logging
logging.basicConfig(
    filename="data/logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# List of scripts to run in order
SCRIPTS = [
    "src/database/setup_database.py",  # Ensure database structure is set
    "src/data_extraction/stock_data_collector.py",  # Fetch most active stocks
    "src/data_extraction/ohlc_data_retriever.py",  # Fetch OHLCV data
    "src/data_processing/data_cleaning.py",  # Clean and validate data
    "src/database/store_stock_metadata.py",  # Fetch stock metadata
    "src/data_extraction/news_sentiment.py",  # Fetch and store news sentiment
    "src/database/load_data.py",  # Load stock data into the database
    "src/database/create_views.py",  # Refresh database views
    "src/database/add_indexes.py",  # Optimize indexes
    "src/database/backup_database.py",  # Backup the database
]


def run_script(script):
    """Runs a Python script and logs the output."""
    console.print(f"[bold yellow]🔄 Running {script}...[/bold yellow]")

    try:
        subprocess.run(["python3", script], check=True)
        console.print(f"[bold green]✅ {script} completed successfully![/bold green]")
        logging.info(f"{script} completed successfully.")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Error in {script}:[/bold red] {e}")
        logging.error(f"Error running {script}: {e}")


def run_pipeline():
    """Runs all scripts in order to refresh stock data."""
    console.print("\n[bold cyan]🚀 Starting the full data pipeline...[/bold cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Processing pipeline...[/bold yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("pipeline", total=len(SCRIPTS))

        for script in SCRIPTS:
            run_script(script)
            progress.update(task, advance=1)

    console.print(
        "\n[bold green]🎉 Data pipeline completed successfully![/bold green]\n"
    )


if __name__ == "__main__":
    run_pipeline()
