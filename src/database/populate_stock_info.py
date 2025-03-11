import logging
import os
import signal
import sys
import time

import psycopg2
import yfinance as yf
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/populate_stock_info.log"
os.makedirs("data/logs", exist_ok=True)

# ✅ Insert 5 blank lines before logging new logs
with open(LOG_FILE, "a") as log_file:
    log_file.write("\n" * 5)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ Database Configuration
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
        logging.error(f"Database connection error: {e}", exc_info=True)
        return None


def fetch_stock_metadata(ticker, max_retries=3):
    """Fetches comprehensive stock metadata from Yahoo Finance with retries."""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                "ticker": ticker,
                "company_name": info.get("longName", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "exchange": info.get("exchange", "Unknown"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "eps": info.get("trailingEps"),
                "earnings_date": info.get("earningsAnnouncement"),
                "ipo_date": info.get("ipoExpectedDate"),
                "price_to_sales_ratio": info.get("priceToSalesTrailing12Months"),
                "price_to_book_ratio": info.get("priceToBook"),
                "enterprise_value": info.get("enterpriseValue"),
                "ebitda": info.get("ebitda"),
                "profit_margin": info.get("profitMargins"),
                "return_on_equity": info.get("returnOnEquity"),
                "beta": info.get("beta"),
                "dividend_yield": info.get("dividendYield"),
            }

        except Exception as e:
            logging.warning(
                f"⚠ Retry {attempt + 1}/{max_retries} - Failed to fetch data for {ticker}: {e}"
            )
            time.sleep(2**attempt)  # Exponential backoff

    logging.error(f"❌ Max retries reached. Skipping {ticker}.")
    return {
        key: None
        for key in [
            "ticker",
            "company_name",
            "sector",
            "industry",
            "exchange",
            "market_cap",
            "pe_ratio",
            "eps",
            "earnings_date",
            "ipo_date",
            "price_to_sales_ratio",
            "price_to_book_ratio",
            "enterprise_value",
            "ebitda",
            "profit_margin",
            "return_on_equity",
            "beta",
            "dividend_yield",
        ]
    }


def populate_stock_info():
    """Fetches stock metadata and populates the stock_info table efficiently."""
    conn = connect_db()
    if not conn:
        return
    cursor = conn.cursor()

    # ✅ Fetch all tickers from the stocks table
    cursor.execute("SELECT DISTINCT ticker FROM stocks;")
    tickers = [row[0] for row in cursor.fetchall()]

    console.print(
        f"\n🚀 Fetching and populating stock metadata for {len(tickers)} stocks...\n",
        style="bold cyan",
    )

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue] Processing {task.fields[ticker]}..."),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("", total=len(tickers), ticker="Starting...")

            batch_data = []
            batch_size = 10  # ✅ Efficient batch insert size

            for ticker in tickers:
                progress.update(
                    task, ticker=ticker
                )  # ✅ Update progress bar with ticker

                metadata = fetch_stock_metadata(ticker)
                batch_data.append(metadata)

                # ✅ Batch insert when batch size is met
                if len(batch_data) >= batch_size:
                    insert_stock_info(cursor, batch_data)
                    conn.commit()
                    batch_data.clear()  # ✅ Reset batch

                progress.advance(task)

            # ✅ Insert remaining records
            if batch_data:
                insert_stock_info(cursor, batch_data)
                conn.commit()

    except KeyboardInterrupt:
        handle_exit(cursor, conn)

    finally:
        cursor.close()
        conn.close()
        console.print(
            "\n[bold green]✅ Stock metadata populated successfully![/bold green]\n"
        )


def insert_stock_info(cursor, batch_data):
    """Performs batch insertion of stock metadata into PostgreSQL."""
    sql = """
        INSERT INTO stock_info (ticker, company_name, sector, industry, exchange, market_cap, pe_ratio, eps, 
                                earnings_date, ipo_date, price_to_sales_ratio, price_to_book_ratio, enterprise_value, 
                                ebitda, profit_margin, return_on_equity, beta, dividend_yield)
        VALUES (%(ticker)s, %(company_name)s, %(sector)s, %(industry)s, %(exchange)s, %(market_cap)s, %(pe_ratio)s, %(eps)s, 
                %(earnings_date)s, %(ipo_date)s, %(price_to_sales_ratio)s, %(price_to_book_ratio)s, %(enterprise_value)s, 
                %(ebitda)s, %(profit_margin)s, %(return_on_equity)s, %(beta)s, %(dividend_yield)s)
        ON CONFLICT (ticker) DO UPDATE
        SET company_name = EXCLUDED.company_name,
            sector = EXCLUDED.sector,
            industry = EXCLUDED.industry,
            exchange = EXCLUDED.exchange,
            market_cap = EXCLUDED.market_cap,
            pe_ratio = EXCLUDED.pe_ratio,
            eps = EXCLUDED.eps,
            earnings_date = EXCLUDED.earnings_date,
            ipo_date = EXCLUDED.ipo_date,
            price_to_sales_ratio = EXCLUDED.price_to_sales_ratio,
            price_to_book_ratio = EXCLUDED.price_to_book_ratio,
            enterprise_value = EXCLUDED.enterprise_value,
            ebitda = EXCLUDED.ebitda,
            profit_margin = EXCLUDED.profit_margin,
            return_on_equity = EXCLUDED.return_on_equity,
            beta = EXCLUDED.beta,
            dividend_yield = EXCLUDED.dividend_yield;
    """

    try:
        cursor.executemany(sql, batch_data)
        logging.info(f"✅ Batch inserted {len(batch_data)} records into stock_info.")
    except Exception as e:
        logging.error(f"❌ Failed batch insert: {e}", exc_info=True)


# ✅ Graceful Exit Handler
def handle_exit(cursor=None, conn=None):
    """Handles graceful shutdown for SIGINT/SIGTERM signals."""
    console.print("\n[bold red]⚠ Process interrupted! Exiting gracefully...[/bold red]")
    logging.info("Process interrupted. Exiting gracefully...")

    if cursor:
        cursor.close()
    if conn:
        conn.close()

    sys.exit(0)


# ✅ Register signal handlers for clean exit
signal.signal(signal.SIGINT, lambda signum, frame: handle_exit())
signal.signal(signal.SIGTERM, lambda signum, frame: handle_exit())


if __name__ == "__main__":
    populate_stock_info()
