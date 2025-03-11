import logging
import os
import subprocess
import sys
from time import sleep

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Setup logging
os.makedirs("data/logs", exist_ok=True)
logging.basicConfig(
    filename="data/logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# List of scripts to run in order
SCRIPTS = [
    ("Setup Database", "src/database/setup_database.py"),
    ("Fetch Most Active Stocks", "src/data_extraction/stock_data_collector.py"),
    ("Fetch OHLCV Data", "src/data_extraction/ohlc_data_retriever.py"),
    ("Calculate Technical Indicators", "src/data_extraction/technical_indicators.py"),
    ("Clean & Validate Data", "src/data_processing/data_cleaning.py"),
    ("Fetch Stock Metadata", "src/database/store_stock_metadata.py"),
    ("Fetch News Sentiment", "src/data_extraction/news_sentiment.py"),
    ("Load Data into Database", "src/database/load_data.py"),
    ("Populate Stock Info", "src/database/populate_stock_info.py"),
    ("Refresh Database Views", "src/database/create_views.py"),
    ("Optimize Indexes", "src/database/add_indexes.py"),
    ("Backup Database", "src/database/backup_database.py"),
]


def run_script(script_name, script_path):
    """Runs a Python script and logs the output."""
    if not os.path.exists(script_path):
        console.print(f"[bold red]❌ {script_name} skipped (File Not Found)[/bold red]")
        logging.error(f"{script_name} skipped - script not found: {script_path}")
        return False

    console.print(f"\n[bold yellow]🔄 Running {script_name}...[/bold yellow]")

    try:
        result = subprocess.run(
            [sys.executable, script_path], check=True, capture_output=True, text=True
        )
        console.print(
            f"[bold green]✅ {script_name} completed successfully![/bold green]"
        )
        logging.info(f"{script_name} completed successfully.\nOutput:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Error in {script_name}:[/bold red] {e.stderr}")
        logging.error(
            f"Error running {script_name}: {e}\nOutput:\n{e.stdout}\nError:\n{e.stderr}"
        )
        return False


def run_pipeline():
    """Runs all scripts in order to refresh stock data, ensuring sequential execution."""
    console.print("\n[bold cyan]🚀 Starting the full data pipeline...[/bold cyan]\n")

    table = Table(title="Pipeline Progress", show_lines=True)
    table.add_column("Step", justify="center", style="bold cyan")
    table.add_column("Script Name", style="bold white")
    table.add_column("Status", justify="center", style="bold")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Processing pipeline...[/bold yellow]"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("pipeline", total=len(SCRIPTS))

        for step, (script_name, script_path) in enumerate(SCRIPTS, start=1):
            success = run_script(script_name, script_path)
            table.add_row(
                str(step),
                script_name,
                "[green]✔ Completed[/green]" if success else "[red]✘ Failed[/red]",
            )

            if not success:
                console.print(
                    f"\n[bold red]❌ Pipeline aborted due to failure in {script_name}.[/bold red]"
                )
                console.print(table)
                return  # Stop execution if a script fails

            progress.update(task, advance=1)
            sleep(1)  # Small delay for better visualization

    console.print(table)
    console.print(
        "\n[bold green]🎉 Data pipeline completed successfully![/bold green]\n"
    )


if __name__ == "__main__":
    run_pipeline()
