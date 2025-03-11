#!/usr/bin/env python3

import logging
import os
import subprocess
import sys

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

# ✅ Setup logging directory
os.makedirs("data/logs", exist_ok=True)

logging.basicConfig(
    filename="data/logs/training_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ List of scripts to run in order
TRAINING_SCRIPTS = [
    ("Fetch Training Data", "src/ml/fetch_training_data.py"),
    ("Data Preprocessing", "src/ml/data_preprocessing.py"),
]


def run_script(script_name, script_path, progress, task):
    """Runs a Python script and logs the output in real-time while updating the progress bar."""
    if not os.path.exists(script_path):
        console.print(f"[bold red]❌ {script_name} skipped (File Not Found)[/bold red]")
        logging.error(f"{script_name} skipped - script not found: {script_path}")
        progress.update(task, advance=1)
        return False

    console.print(f"\n[bold yellow]🔄 Running {script_name}...[/bold yellow]")

    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # ✅ Ensures real-time streaming
            universal_newlines=True,
        )

        # ✅ Live stream output while updating progress
        for line in process.stdout:
            console.print(f"[white]{line.strip()}[/white]")
            progress.update(task, advance=0.1)  # ✅ Update progress slightly

        stdout, stderr = process.communicate()
        exit_code = process.returncode

        if exit_code == 0:
            console.print(
                f"[bold green]✅ {script_name} completed successfully![/bold green]"
            )
            logging.info(f"{script_name} completed successfully.\nOutput:\n{stdout}")
            progress.update(task, advance=1)
            return True
        else:
            console.print(f"[bold red]❌ Error in {script_name}:[/bold red] {stderr}")
            logging.error(f"Error running {script_name}: {stderr}\nOutput:\n{stdout}")
            progress.update(task, advance=1)
            return False

    except Exception as e:
        console.print(
            f"[bold red]❌ Unexpected error in {script_name}:[/bold red] {str(e)}"
        )
        logging.error(f"Unexpected error in {script_name}: {str(e)}")
        progress.update(task, advance=1)
        return False


def run_training_pipeline():
    """Runs all scripts sequentially with a working progress bar."""
    console.print(
        "\n[bold cyan]🚀 Starting the model training pipeline...[/bold cyan]\n"
    )

    table = Table(title="Training Pipeline Progress", show_lines=True)
    table.add_column("Step", justify="center", style="bold cyan")
    table.add_column("Script Name", style="bold white")
    table.add_column("Status", justify="center", style="bold")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Processing training pipeline...[/bold yellow]"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training Pipeline", total=len(TRAINING_SCRIPTS))

        for step, (script_name, script_path) in enumerate(TRAINING_SCRIPTS, start=1):
            success = run_script(script_name, script_path, progress, task)

            table.add_row(
                str(step),
                script_name,
                "[green]✔ Completed[/green]" if success else "[red]✘ Failed[/red]",
            )

            if not success:
                console.print(
                    f"\n[bold red]❌ Training pipeline aborted due to failure in {script_name}.[/bold red]"
                )
                console.print(table)
                sys.exit(1)  # Exit immediately on failure

        progress.update(task, advance=1)

    console.print(table)
    console.print(
        "\n[bold green]🎉 Model training pipeline completed successfully![/bold green]\n"
    )


if __name__ == "__main__":
    run_training_pipeline()
