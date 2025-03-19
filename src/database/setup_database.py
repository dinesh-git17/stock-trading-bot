import logging
import os
import signal
import sys

from rich.console import Console
from sqlalchemy import text

# ✅ Import utilities
from src.tools.utils import get_database_engine, handle_exceptions, setup_logging

# ✅ Setup Logging
LOG_FILE = "data/logs/database_setup.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)
console = Console()
logger.info("🚀 Logging setup complete.")


@handle_exceptions
def update_stock_info_table(engine):
    """Ensures `stock_info` table has all required columns."""
    console.print(
        "[bold yellow]🔄 Checking and updating `stock_info` table...[/bold yellow]"
    )
    required_columns = {
        "ticker": "VARCHAR(10) PRIMARY KEY",
        "company_name": "TEXT",
        "sector": "TEXT",
        "industry": "TEXT",
        "exchange": "TEXT",
        "market_cap": "BIGINT",
        "pe_ratio": "NUMERIC",
        "eps": "NUMERIC",
        "earnings_date": "DATE",
        "ipo_date": "DATE",
        "price_to_sales_ratio": "NUMERIC",
        "price_to_book_ratio": "NUMERIC",
        "enterprise_value": "BIGINT",
        "ebitda": "BIGINT",
        "profit_margin": "NUMERIC",
        "return_on_equity": "NUMERIC",
        "beta": "NUMERIC",
        "dividend_yield": "NUMERIC",
    }

    with engine.connect() as conn:
        existing_columns = {
            row[0]
            for row in conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = 'stock_info';"
                )
            )
        }

        for column, column_type in required_columns.items():
            if column not in existing_columns:
                conn.execute(
                    text(f"ALTER TABLE stock_info ADD COLUMN {column} {column_type};")
                )
                console.print(f"[bold green]➕ Added column:[/bold green] {column}")

        for column in existing_columns:
            if column not in required_columns:
                conn.execute(
                    text(f"ALTER TABLE stock_info DROP COLUMN {column} CASCADE;")
                )
                console.print(
                    f"[bold red]❌ Removed outdated column:[/bold red] {column}"
                )

    console.print(
        "[bold green]✅ `stock_info` table updated successfully![/bold green]"
    )
    logger.info("Updated `stock_info` table with latest schema.")


@handle_exceptions
def create_tables():
    """Creates all necessary tables in the PostgreSQL database."""
    engine = get_database_engine()

    with engine.connect() as conn:
        try:
            conn.execute(
                text(
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
                    adjusted_close NUMERIC,
                    UNIQUE (ticker, date)
                );
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS stock_info (
                    ticker VARCHAR(10) PRIMARY KEY,
                    company_name TEXT,
                    sector TEXT,
                    industry TEXT,
                    exchange TEXT,
                    market_cap BIGINT,
                    pe_ratio NUMERIC,
                    eps NUMERIC,
                    earnings_date DATE,
                    ipo_date DATE,
                    price_to_sales_ratio NUMERIC,
                    price_to_book_ratio NUMERIC,
                    enterprise_value BIGINT,
                    ebitda BIGINT,
                    profit_margin NUMERIC,
                    return_on_equity NUMERIC,
                    beta NUMERIC,
                    dividend_yield NUMERIC
                );
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    source_name TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    url TEXT NOT NULL UNIQUE,
                    sentiment_score NUMERIC,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    date TIMESTAMP NOT NULL,
                    sma_50 NUMERIC,
                    sma_200 NUMERIC,
                    ema_50 NUMERIC,
                    ema_200 NUMERIC,
                    rsi_14 NUMERIC,
                    adx_14 NUMERIC,
                    atr_14 NUMERIC,
                    cci_20 NUMERIC,
                    williamsr_14 NUMERIC,
                    macd NUMERIC,
                    macd_signal NUMERIC,
                    macd_hist NUMERIC,
                    bb_upper NUMERIC,
                    bb_middle NUMERIC,
                    bb_lower NUMERIC,
                    stoch_k NUMERIC,
                    stoch_d NUMERIC,
                    FOREIGN KEY (ticker, date) REFERENCES stocks (ticker, date) ON DELETE CASCADE,
                    UNIQUE (ticker, date)
                );
            """
                )
            )

            update_stock_info_table(engine)
            console.print(
                "[bold green]✅ Database tables created successfully![/bold green]"
            )
            logger.info("Database tables created successfully.")
        except Exception as e:
            console.print(f"[bold red]❌ Error creating tables:[/bold red] {e}")
            logger.error(f"Error creating tables: {e}", exc_info=True)


signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

if __name__ == "__main__":
    create_tables()
