import logging
import os

import psycopg2
from dotenv import load_dotenv
from rich.console import Console

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/database_views.log"
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


def create_views():
    """Creates SQL views and materialized views for optimized data access."""
    conn = connect_db()
    if not conn:
        return

    with conn.cursor() as cur:
        try:
            console.print("[bold yellow]🔄 Dropping old views...[/bold yellow]")
            cur.execute("DROP VIEW IF EXISTS latest_stock_data CASCADE;")
            cur.execute("DROP VIEW IF EXISTS high_momentum_stocks CASCADE;")
            cur.execute("DROP VIEW IF EXISTS oversold_stocks CASCADE;")
            cur.execute("DROP VIEW IF EXISTS market_trend_analysis CASCADE;")
            cur.execute("DROP VIEW IF EXISTS stock_sector_analysis CASCADE;")

            cur.execute(
                "DROP MATERIALIZED VIEW IF EXISTS materialized_stock_analysis CASCADE;"
            )
            cur.execute(
                "DROP MATERIALIZED VIEW IF EXISTS materialized_sector_performance CASCADE;"
            )

            console.print("[bold blue]🔄 Creating new views...[/bold blue]")

            # ✅ View: Latest Stock Data (Real-time Data)
            cur.execute(
                """
                CREATE VIEW latest_stock_data AS
                SELECT DISTINCT ON (s.ticker) s.*
                FROM stocks s
                ORDER BY s.ticker, s.date DESC;
            """
            )

            # ✅ View: High Momentum Stocks (Identifies MACD & RSI crossovers)
            cur.execute(
                """
                CREATE VIEW high_momentum_stocks AS
                SELECT s.ticker, s.date, s.close, ti.rsi_14, ti.macd, ti.macd_signal
                FROM stocks s
                JOIN technical_indicators ti ON s.ticker = ti.ticker AND s.date = ti.date
                WHERE ti.rsi_14 > 50 AND ti.macd > ti.macd_signal
                ORDER BY ti.rsi_14 DESC;
            """
            )

            # ✅ View: Oversold Stocks (RSI below 30 - Potential Buy Signals)
            cur.execute(
                """
                CREATE VIEW oversold_stocks AS
                SELECT s.ticker, s.date, s.close, ti.rsi_14
                FROM stocks s
                JOIN technical_indicators ti ON s.ticker = ti.ticker AND s.date = ti.date
                WHERE ti.rsi_14 < 30
                ORDER BY ti.rsi_14 ASC;
            """
            )

            # ✅ View: Market Trend Analysis (Aggregates Trends)
            cur.execute(
                """
                CREATE VIEW market_trend_analysis AS
                SELECT s.ticker, 
                       MIN(s.close) AS min_price, 
                       MAX(s.close) AS max_price, 
                       AVG(s.close) AS avg_price, 
                       COUNT(s.date) AS days_tracked
                FROM stocks s
                WHERE s.date >= CURRENT_DATE - INTERVAL '90 days'
                GROUP BY s.ticker
                ORDER BY avg_price DESC;
            """
            )

            # ✅ View: Stock Sector Performance
            cur.execute(
                """
                CREATE VIEW stock_sector_analysis AS
                SELECT si.sector, 
                       COUNT(s.ticker) AS total_stocks, 
                       AVG(s.close) AS avg_sector_price
                FROM stocks s
                JOIN stock_info si ON s.ticker = si.ticker
                GROUP BY si.sector
                ORDER BY avg_sector_price DESC;
            """
            )

            console.print(
                "[bold green]✅ Standard views created successfully![/bold green]"
            )

            console.print(
                "[bold yellow]🔄 Creating materialized views...[/bold yellow]"
            )

            # ✅ Materialized View: Cached Stock Analysis
            cur.execute(
                """
                CREATE MATERIALIZED VIEW materialized_stock_analysis AS
                SELECT s.ticker, s.date, s.open, s.high, s.low, s.close, s.volume,
                       ti.sma_50, ti.ema_50, ti.rsi_14, ti.macd, ti.macd_signal,
                       si.market_cap, si.pe_ratio, si.earnings_date
                FROM stocks s
                JOIN technical_indicators ti ON s.ticker = ti.ticker AND s.date = ti.date
                JOIN stock_info si ON s.ticker = si.ticker
                ORDER BY s.ticker, s.date;
            """
            )

            # ✅ Materialized View: Sector Performance Analysis
            cur.execute(
                """
                CREATE MATERIALIZED VIEW materialized_sector_performance AS
                SELECT si.sector, 
                       COUNT(s.ticker) AS total_stocks, 
                       SUM(si.market_cap) AS total_market_cap, 
                       AVG(s.close) AS avg_sector_price
                FROM stocks s
                JOIN stock_info si ON s.ticker = si.ticker
                GROUP BY si.sector
                ORDER BY total_market_cap DESC;
            """
            )

            console.print(
                "[bold green]✅ Materialized views created successfully![/bold green]"
            )
            logging.info("Materialized views created successfully.")

        except Exception as e:
            console.print(f"[bold red]❌ Error creating SQL views:[/bold red] {e}")
            logging.error(f"Error creating SQL views: {e}")

    conn.close()


def refresh_materialized_views():
    """Refreshes materialized views for up-to-date stock analysis."""
    conn = connect_db()
    if not conn:
        return

    with conn.cursor() as cur:
        try:
            console.print("[bold blue]🔄 Refreshing materialized views...[/bold blue]")
            cur.execute("REFRESH MATERIALIZED VIEW materialized_stock_analysis;")
            cur.execute("REFRESH MATERIALIZED VIEW materialized_sector_performance;")
            console.print(
                "[bold green]✅ Materialized views refreshed successfully![/bold green]"
            )
            logging.info("Materialized views refreshed successfully.")

        except Exception as e:
            console.print(
                f"[bold red]❌ Error refreshing materialized views:[/bold red] {e}"
            )
            logging.error(f"Error refreshing materialized views: {e}")

    conn.close()


if __name__ == "__main__":
    create_views()
    refresh_materialized_views()
