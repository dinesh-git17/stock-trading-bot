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
    filename="data/logs/stock_metadata.log",
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


def fetch_stock_metadata(ticker):
    """Fetches market cap, P/E ratio, earnings date, and financial metrics from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "company_name": info.get("longName", "Unknown"),
            "sector": info.get("sector", "Unknown"),
            "market_cap": info.get("marketCap", None),
            "pe_ratio": info.get("trailingPE", None),
            "eps": info.get("trailingEps", None),
            "earnings_date": info.get("nextEarningsDate", None),
        }
    except Exception as e:
        logging.warning(f"Failed to fetch metadata for {ticker}: {e}")
        return None


def store_stock_metadata():
    """Fetches and updates stock metadata in the `stock_info` table."""
    conn = connect_db()
    if not conn:
        return
    cursor = conn.cursor()

    # Fetch all tickers from the `stocks` table
    cursor.execute("SELECT DISTINCT ticker FROM stocks;")
    tickers = [row[0] for row in cursor.fetchall()]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Fetching stock metadata...[/bold yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("metadata", total=len(tickers))

        for ticker in tickers:
            metadata = fetch_stock_metadata(ticker)
            if metadata:
                try:
                    cursor.execute(
                        """
                        INSERT INTO stock_info (ticker, company_name, sector, market_cap, pe_ratio, eps, earnings_date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker) DO UPDATE
                        SET company_name = EXCLUDED.company_name,
                            sector = EXCLUDED.sector,
                            market_cap = EXCLUDED.market_cap,
                            pe_ratio = EXCLUDED.pe_ratio,
                            eps = EXCLUDED.eps,
                            earnings_date = EXCLUDED.earnings_date;
                        """,
                        (
                            ticker,
                            metadata["company_name"],
                            metadata["sector"],
                            metadata["market_cap"],
                            metadata["pe_ratio"],
                            metadata["eps"],
                            metadata["earnings_date"],
                        ),
                    )
                    conn.commit()
                    logging.info(f"Updated stock metadata for {ticker}")

                except Exception as e:
                    logging.error(f"Failed to update stock metadata for {ticker}: {e}")
                    conn.rollback()

            progress.update(task, advance=1)

    cursor.close()
    conn.close()
    console.print("[bold green]✅ Stock metadata updated successfully![/bold green]")


if __name__ == "__main__":
    store_stock_metadata()
