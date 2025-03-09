import logging
import os
import subprocess
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/database_backup.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# Database Configuration
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Backup Directory
BACKUP_DIR = "data/backups/"
os.makedirs(BACKUP_DIR, exist_ok=True)


def backup_database():
    """
    Creates a PostgreSQL database backup and stores it in the backup directory.
    """
    console.print("\n[bold yellow]🔄 Creating database backup...[/bold yellow]")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_file = os.path.join(BACKUP_DIR, f"backup_{timestamp}.sql")

    env = os.environ.copy()
    env["PGPASSWORD"] = DB_PASSWORD  # Pass password securely

    command = [
        "pg_dump",
        "-h",
        DB_HOST,
        "-p",
        DB_PORT,
        "-U",
        DB_USER,
        "-F",
        "c",  # Custom format for better compression
        "-f",
        backup_file,
        DB_NAME,
    ]

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]Backing up database...[/bold yellow]"),
            console=console,
        ) as progress:
            task = progress.add_task("backup", total=None)
            subprocess.run(command, check=True, env=env)
            progress.update(task, completed=1)

        console.print(
            f"[bold green]✅ Backup created successfully: {backup_file}[/bold green]"
        )
        logging.info(f"Backup created: {backup_file}")

    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Backup failed:[/bold red] {e}")
        logging.error(f"Backup failed: {e}")


if __name__ == "__main__":
    backup_database()
