import logging
import os

import psycopg2
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/database_views.log",
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


def create_views():
    """Creates SQL views for optimized data access."""
    conn = connect_db()
    if not conn:
        return

    with conn.cursor() as cur:
        try:
            # ✅ Drop all views using CASCADE to remove dependencies
            cur.execute("DROP VIEW IF EXISTS stock_sector_analysis CASCADE;")
            cur.execute("DROP VIEW IF EXISTS stock_filtered_by_timeframe CASCADE;")
            cur.execute("DROP VIEW IF EXISTS stock_analysis_view CASCADE;")

            # View 1: General Stock Analysis
            cur.execute(
                """
                CREATE VIEW stock_analysis_view AS
                SELECT
                    s.ticker,
                    s.date,
                    s.open,
                    s.high,
                    s.low,
                    s.close,
                    s.volume,
                    s.dividends,
                    s.stock_splits,
                    ti.sma_20,
                    ti.ema_20,
                    ti.rsi_14,
                    ti.macd,
                    ti.macd_signal,
                    ti.macd_hist,
                    ti.bb_upper,
                    ti.bb_middle,
                    ti.bb_lower
                FROM stocks s
                LEFT JOIN technical_indicators ti
                ON s.ticker = ti.ticker AND s.date = ti.date
                ORDER BY s.ticker, s.date;
            """
            )

            # View 2: Filter by Timeframe (Last 30 Days)
            cur.execute(
                """
                CREATE VIEW stock_filtered_by_timeframe AS
                SELECT *
                FROM stock_analysis_view
                WHERE date >= CURRENT_DATE - INTERVAL '30 days';
            """
            )

            # View 3: Stock Sector Analysis
            cur.execute(
                """
                CREATE VIEW stock_sector_analysis AS
                SELECT s.*, si.sector
                FROM stocks s
                JOIN stock_info si ON s.ticker = si.ticker
                ORDER BY si.sector, s.ticker, s.date;
            """
            )

            console.print(
                "[bold green]✅ Enhanced SQL views created successfully![/bold green]"
            )
            logging.info("Enhanced SQL views created successfully.")

        except Exception as e:
            console.print(f"[bold red]❌ Error creating SQL views:[/bold red] {e}")
            logging.error(f"Error creating SQL views: {e}")

    conn.close()


if __name__ == "__main__":
    create_views()
