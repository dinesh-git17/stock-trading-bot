import os
import psycopg2
import logging
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/database_setup.log",
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
    """Connect to the PostgreSQL database."""
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


def update_stock_info_table(cursor):
    """Adds missing columns to the stock_info table if they do not exist."""
    console.print(
        "[bold yellow]🔄 Checking and updating stock_info table...[/bold yellow]"
    )

    alter_queries = [
        "ALTER TABLE stock_info ADD COLUMN IF NOT EXISTS market_cap BIGINT;",
        "ALTER TABLE stock_info ADD COLUMN IF NOT EXISTS pe_ratio NUMERIC;",
        "ALTER TABLE stock_info ADD COLUMN IF NOT EXISTS eps NUMERIC;",
        "ALTER TABLE stock_info ADD COLUMN IF NOT EXISTS earnings_date DATE;",
    ]

    for query in alter_queries:
        cursor.execute(query)

    console.print("[bold green]✅ stock_info table updated successfully![/bold green]")
    logging.info("Updated stock_info table to include missing columns.")


def create_tables():
    """Creates necessary tables including stock_info for sector filtering."""
    conn = connect_db()
    if not conn:
        return

    with conn.cursor() as cur:
        try:
            # Create stocks table (OHLCV data)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS stocks (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    date TIMESTAMP NOT NULL,
                    open NUMERIC,
                    high NUMERIC,
                    low NUMERIC,
                    close NUMERIC,
                    volume BIGINT,
                    dividends NUMERIC,
                    stock_splits NUMERIC,
                    UNIQUE (ticker, date)
                );
            """
            )

            # Create technical indicators table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    date TIMESTAMP NOT NULL,
                    sma_20 NUMERIC,
                    ema_20 NUMERIC,
                    rsi_14 NUMERIC,
                    macd NUMERIC,
                    macd_signal NUMERIC,
                    macd_hist NUMERIC,
                    bb_upper NUMERIC,
                    bb_middle NUMERIC,
                    bb_lower NUMERIC,
                    FOREIGN KEY (ticker, date) REFERENCES stocks (ticker, date) ON DELETE CASCADE,
                    UNIQUE (ticker, date)
                );
            """
            )

            # ✅ Create stock_info table (stores metadata like sector)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_info (
                    ticker VARCHAR(10) PRIMARY KEY,
                    company_name TEXT,
                    sector TEXT,
                    market_cap BIGINT,
                    pe_ratio NUMERIC,
                    eps NUMERIC,
                    earnings_date DATE
                );
            """
            )

            # Create news_sentiment table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    title TEXT,
                    sentiment_score NUMERIC,
                    UNIQUE (ticker, published_at)
                );
            """
            )

            update_stock_info_table(cur)
            console.print(
                "[bold green]✅ Database tables created successfully![/bold green]"
            )
            logging.info("Database tables created successfully.")

        except Exception as e:
            console.print(f"[bold red]❌ Error creating tables:[/bold red] {e}")
            logging.error(f"Error creating tables: {e}")

    conn.close()


if __name__ == "__main__":
    create_tables()
