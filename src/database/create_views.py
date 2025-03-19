import logging
import os

from dotenv import load_dotenv
from rich.console import Console
from sqlalchemy import text

from src.tools.utils import get_database_engine, handle_exceptions, setup_logging

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/database_views.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)

# ✅ Setup Console
console = Console()


@handle_exceptions
def drop_views(engine):
    """Drops existing views and materialized views to ensure fresh creation."""
    drop_queries = [
        "DROP VIEW IF EXISTS latest_stock_data CASCADE;",
        "DROP VIEW IF EXISTS high_momentum_stocks CASCADE;",
        "DROP VIEW IF EXISTS oversold_stocks CASCADE;",
        "DROP VIEW IF EXISTS market_trend_analysis CASCADE;",
        "DROP VIEW IF EXISTS stock_sector_analysis CASCADE;",
        "DROP MATERIALIZED VIEW IF EXISTS materialized_stock_analysis CASCADE;",
        "DROP MATERIALIZED VIEW IF EXISTS materialized_sector_performance CASCADE;",
    ]

    with engine.begin() as connection:
        for query in drop_queries:
            connection.execute(text(query))

    console.print("[bold yellow]🔄 Dropped old views successfully![/bold yellow]")
    logger.info("Dropped old views successfully.")


@handle_exceptions
def create_standard_views(engine):
    """Creates standard SQL views for efficient data analysis."""
    view_queries = {
        "latest_stock_data": """
            CREATE VIEW latest_stock_data AS
            SELECT DISTINCT ON (s.ticker) s.*
            FROM stocks s
            ORDER BY s.ticker, s.date DESC;
        """,
        "high_momentum_stocks": """
            CREATE VIEW high_momentum_stocks AS
            SELECT s.ticker, s.date, s.close, ti.rsi_14, ti.macd, ti.macd_signal
            FROM stocks s
            JOIN technical_indicators ti ON s.ticker = ti.ticker AND s.date = ti.date
            WHERE ti.rsi_14 > 50 AND ti.macd > ti.macd_signal
            ORDER BY ti.rsi_14 DESC;
        """,
        "oversold_stocks": """
            CREATE VIEW oversold_stocks AS
            SELECT s.ticker, s.date, s.close, ti.rsi_14
            FROM stocks s
            JOIN technical_indicators ti ON s.ticker = ti.ticker AND s.date = ti.date
            WHERE ti.rsi_14 < 30
            ORDER BY ti.rsi_14 ASC;
        """,
        "market_trend_analysis": """
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
        """,
        "stock_sector_analysis": """
            CREATE VIEW stock_sector_analysis AS
            SELECT si.sector, 
                   COUNT(s.ticker) AS total_stocks, 
                   AVG(s.close) AS avg_sector_price
            FROM stocks s
            JOIN stock_info si ON s.ticker = si.ticker
            GROUP BY si.sector
            ORDER BY avg_sector_price DESC;
        """,
    }

    with engine.begin() as connection:
        for view_name, query in view_queries.items():
            connection.execute(text(query))

    console.print("[bold green]✅ Standard views created successfully![/bold green]")
    logger.info("Standard views created successfully.")


@handle_exceptions
def create_materialized_views(engine):
    """Creates materialized views for improved query performance."""
    materialized_view_queries = {
        "materialized_stock_analysis": """
            CREATE MATERIALIZED VIEW materialized_stock_analysis AS
            SELECT s.ticker, s.date, s.open, s.high, s.low, s.close, s.volume,
                   ti.sma_50, ti.ema_50, ti.rsi_14, ti.macd, ti.macd_signal,
                   si.market_cap, si.pe_ratio, si.earnings_date
            FROM stocks s
            JOIN technical_indicators ti ON s.ticker = ti.ticker AND s.date = ti.date
            JOIN stock_info si ON s.ticker = si.ticker
            ORDER BY s.ticker, s.date;
        """,
        "materialized_sector_performance": """
            CREATE MATERIALIZED VIEW materialized_sector_performance AS
            SELECT si.sector, 
                   COUNT(s.ticker) AS total_stocks, 
                   SUM(si.market_cap) AS total_market_cap, 
                   AVG(s.close) AS avg_sector_price
            FROM stocks s
            JOIN stock_info si ON s.ticker = si.ticker
            GROUP BY si.sector
            ORDER BY total_market_cap DESC;
        """,
    }

    with engine.begin() as connection:
        for view_name, query in materialized_view_queries.items():
            connection.execute(text(query))

    console.print(
        "[bold green]✅ Materialized views created successfully![/bold green]"
    )
    logger.info("Materialized views created successfully.")


@handle_exceptions
def refresh_materialized_views(engine):
    """Refreshes materialized views to keep data up to date."""
    refresh_queries = [
        "REFRESH MATERIALIZED VIEW materialized_stock_analysis;",
        "REFRESH MATERIALIZED VIEW materialized_sector_performance;",
    ]

    with engine.begin() as connection:
        for query in refresh_queries:
            connection.execute(text(query))

    console.print(
        "[bold green]✅ Materialized views refreshed successfully![/bold green]"
    )
    logger.info("Materialized views refreshed successfully.")


def create_and_refresh_views():
    """Main function to create and refresh views."""
    engine = get_database_engine()
    if not engine:
        return

    drop_views(engine)
    create_standard_views(engine)
    create_materialized_views(engine)
    refresh_materialized_views(engine)


if __name__ == "__main__":
    create_and_refresh_views()
