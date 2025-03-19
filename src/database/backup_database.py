import logging
import os
import subprocess
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.tools.utils import handle_exceptions, setup_logging

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/database_backup.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)

# ✅ Setup Console
console = Console()

# ✅ Backup Directory
BACKUP_DIR = "data/backups/"
os.makedirs(BACKUP_DIR, exist_ok=True)


@handle_exceptions
def backup_database():
    """
    Creates a compressed PostgreSQL database backup and stores it in the backup directory.
    """
    console.print("\n[bold yellow]🔄 Creating database backup...[/bold yellow]")

    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")

    if not all([db_name, db_user, db_password, db_host, db_port]):
        console.print(
            "[bold red]❌ Missing database credentials in .env file![/bold red]"
        )
        logger.error("Database backup failed: Missing credentials in .env file.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_file = os.path.join(BACKUP_DIR, f"backup_{timestamp}.sql.gz")

    env = os.environ.copy()
    if db_password is None:
        console.print("[bold red]❌ Missing database password in .env file![/bold red]")
        logger.error("Database backup failed: Missing database password in .env file.")
        return

    env["PGPASSWORD"] = db_password

    command = [
        "pg_dump",
        "-h",
        db_host,
        "-p",
        db_port,
        "-U",
        db_user,
        "-F",
        "c",  # ✅ Custom format for better compression
        "-f",
        "-",  # ✅ Output to stdout for piping
        db_name,
    ]

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]Backing up database...[/bold yellow]"),
            console=console,
        ) as progress:
            task = progress.add_task("backup", total=None)

            # ✅ Run backup and compress output using gzip
            with open(backup_file, "wb") as f:
                subprocess.run(command, check=True, env=env, stdout=subprocess.PIPE)
                subprocess.run(
                    ["gzip"], check=True, env=env, stdin=subprocess.PIPE, stdout=f
                )

            progress.update(task, completed=1)

        console.print(
            f"[bold green]✅ Backup created successfully: {backup_file}[/bold green]"
        )
        logger.info(f"Backup created: {backup_file}")

    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Backup failed:[/bold red] {e}")
        logger.error(f"Backup failed: {e}", exc_info=True)


if __name__ == "__main__":
    backup_database()
