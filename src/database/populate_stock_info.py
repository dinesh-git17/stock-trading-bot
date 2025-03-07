import os
import psycopg2
import yfinance as yf
import logging
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/populate_stock_info.log",
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
        return conn
    except Exception as e:
        console.print(f"[bold red]❌ Database connection error:[/bold red] {e}")
        logging.error(f"Database connection error: {e}")
        return None


def fetch_sector_data(ticker):
    """Fetches the sector and company name from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get("longName", "Unknown"), info.get("sector", "Unknown")
    except Exception as e:
        logging.warning(f"Failed to fetch sector for {ticker}: {e}")
        return "Unknown", "Unknown"


def populate_stock_info():
    """Fetches sector data and populates the stock_info table."""
    conn = connect_db()
    if not conn:
        return
    cursor = conn.cursor()

    # Fetch all tickers from the stocks table
    cursor.execute("SELECT DISTINCT ticker FROM stocks;")
    tickers = [row[0] for row in cursor.fetchall()]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Fetching sector data...[/bold yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("fetching", total=len(tickers))

        for ticker in tickers:
            company_name, sector = fetch_sector_data(ticker)

            try:
                cursor.execute(
                    """
                    INSERT INTO stock_info (ticker, company_name, sector)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (ticker) DO UPDATE
                    SET company_name = EXCLUDED.company_name, sector = EXCLUDED.sector;
                    """,
                    (ticker, company_name, sector),
                )
                conn.commit()
                logging.info(
                    f"Added {ticker} - {company_name} ({sector}) to stock_info."
                )

            except Exception as e:
                logging.error(f"Failed to insert {ticker} sector data: {e}")
                conn.rollback()

            progress.update(task, advance=1)

    cursor.close()
    conn.close()
    console.print("[bold green]✅ Sector data populated successfully![/bold green]")


if __name__ == "__main__":
    populate_stock_info()
