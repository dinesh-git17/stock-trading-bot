import logging
import os

from dotenv import load_dotenv
from rich.console import Console
from sqlalchemy import text

from src.tools.utils import get_database_engine, handle_exceptions, setup_logging

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/database_indexes.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)

# ✅ Setup Console
console = Console()


@handle_exceptions
def add_indexes(engine):
    """Adds indexes to optimize query performance for trading bot analytics."""
    index_queries = [
        # ✅ Multi-Column Index: Optimizes ticker-based queries & joins
        "CREATE INDEX IF NOT EXISTS idx_stocks_ticker_date ON stocks (ticker, date DESC);",
        "CREATE INDEX IF NOT EXISTS idx_technical_ticker_date ON technical_indicators (ticker, date DESC);",
        # ✅ Sector-Based Indexing (Speeds up sector-based filtering)
        "CREATE INDEX IF NOT EXISTS idx_stock_info_sector ON stock_info (sector);",
        # ✅ Index on Market Cap (Speeds up sorting for largest companies)
        "CREATE INDEX IF NOT EXISTS idx_stock_info_marketcap ON stock_info (market_cap DESC);",
        # ✅ Index on PE Ratio (For filtering undervalued stocks)
        "CREATE INDEX IF NOT EXISTS idx_stock_info_pe_ratio ON stock_info (pe_ratio);",
        # ✅ Index on RSI for fast overbought/oversold queries
        "CREATE INDEX IF NOT EXISTS idx_technical_rsi ON technical_indicators (rsi_14);",
        # ✅ Index on MACD Signal (Speeds up momentum trading analysis)
        "CREATE INDEX IF NOT EXISTS idx_technical_macd ON technical_indicators (macd_signal);",
        # ✅ Full-Text Search Index on News Sentiment (Speeds up text searches)
        "CREATE INDEX IF NOT EXISTS idx_news_sentiment_title ON news_sentiment USING GIN (to_tsvector('english', title));",
    ]

    with engine.begin() as connection:
        for query in index_queries:
            connection.execute(text(query))

    console.print("[bold green]✅ Indexes added successfully![/bold green]")
    logger.info("Indexes added successfully.")


def main():
    """Main function to add indexes to the database."""
    engine = get_database_engine()
    if not engine:
        return

    add_indexes(engine)


if __name__ == "__main__":
    main()
