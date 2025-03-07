import os
import psycopg2
import logging
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/database_indexes.log",
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


def connect_db():
    """Connects to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        console.print(f"[bold red]❌ Database connection error:[/bold red] {e}")
        logging.error(f"Database connection error: {e}")
        return None


def add_indexes():
    """Adds indexes to optimize query performance."""
    conn = connect_db()
    if not conn:
        return

    with conn.cursor() as cur:
        try:
            console.print(
                "[bold yellow]🔄 Adding indexes to optimize database queries...[/bold yellow]"
            )

            # ✅ Index on stocks.ticker and stocks.date (speeds up lookups & joins)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_stocks_ticker_date ON stocks (ticker, date);"
            )

            # ✅ Index on technical_indicators.ticker and technical_indicators.date (for joins)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_technical_ticker_date ON technical_indicators (ticker, date);"
            )

            # ✅ Index on stock_info.ticker (for sector-based queries)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_stock_info_ticker ON stock_info (ticker);"
            )

            # ✅ Index on stock_info.sector (to speed up filtering by sector)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_stock_info_sector ON stock_info (sector);"
            )

            console.print("[bold green]✅ Indexes added successfully![/bold green]")
            logging.info("Indexes added successfully.")

        except Exception as e:
            console.print(f"[bold red]❌ Error adding indexes:[/bold red] {e}")
            logging.error(f"Error adding indexes: {e}")

    conn.close()


if __name__ == "__main__":
    add_indexes()
