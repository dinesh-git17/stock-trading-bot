import logging
import os

import psycopg2
from dotenv import load_dotenv
from rich.console import Console

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/database_indexes.log"
os.makedirs("data/logs", exist_ok=True)

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
        conn.autocommit = True
        return conn
    except Exception as e:
        console.print(f"[bold red]❌ Database connection error:[/bold red] {e}")
        logging.error(f"Database connection error: {e}", exc_info=True)
        return None


def add_indexes():
    """Adds indexes to optimize query performance for trading bot analytics."""
    conn = connect_db()
    if not conn:
        return

    with conn.cursor() as cur:
        try:
            console.print(
                "[bold yellow]🔄 Adding indexes to optimize database queries...[/bold yellow]"
            )

            # ✅ Multi-Column Index: Optimizes ticker-based queries & joins
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_stocks_ticker_date ON stocks (ticker, date DESC);"
            )

            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_technical_ticker_date ON technical_indicators (ticker, date DESC);"
            )

            # ✅ Sector-Based Indexing (Speeds up sector-based filtering)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_stock_info_sector ON stock_info (sector);"
            )

            # ✅ Index on Market Cap (Speeds up sorting for largest companies)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_stock_info_marketcap ON stock_info (market_cap DESC);"
            )

            # ✅ Index on PE Ratio (For filtering undervalued stocks)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_stock_info_pe_ratio ON stock_info (pe_ratio);"
            )

            # ✅ Index on RSI for fast overbought/oversold queries
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_technical_rsi ON technical_indicators (rsi_14);"
            )

            # ✅ Index on MACD Signal (Speeds up momentum trading analysis)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_technical_macd ON technical_indicators (macd_signal);"
            )

            # ✅ Full-Text Search Index on News Sentiment (Speeds up text searches)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_news_sentiment_title ON news_sentiment USING GIN (to_tsvector('english', title));"
            )

            console.print("[bold green]✅ Indexes added successfully![/bold green]")
            logging.info("Indexes added successfully.")

        except Exception as e:
            console.print(f"[bold red]❌ Error adding indexes:[/bold red] {e}")
            logging.error(f"Error adding indexes: {e}", exc_info=True)

    conn.close()


if __name__ == "__main__":
    add_indexes()
