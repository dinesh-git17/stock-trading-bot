import logging
import os

import yfinance as yf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from sqlalchemy import text

# ✅ Import utilities
from src.tools.utils import get_database_engine, handle_exceptions, setup_logging

# ✅ Setup Logging
LOG_FILE = "data/logs/stock_metadata.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)
console = Console()
logger.info("🚀 Logging setup complete.")


@handle_exceptions
def fetch_stock_metadata(ticker):
    """Fetches market cap, P/E ratio, earnings date, and financial metrics from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "company_name": info.get("longName", "Unknown"),
            "sector": info.get("sector", "Unknown"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "earnings_date": info.get("nextEarningsDate"),
        }
    except Exception as e:
        logger.warning(f"Failed to fetch metadata for {ticker}: {e}")
        return None


@handle_exceptions
def store_stock_metadata():
    """Fetches and updates stock metadata in the `stock_info` table."""
    engine = get_database_engine()

    with engine.connect() as conn:
        tickers = [
            row[0] for row in conn.execute(text("SELECT DISTINCT ticker FROM stocks;"))
        ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Fetching stock metadata...[/bold yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("metadata", total=len(tickers))

        with engine.begin() as conn:
            for ticker in tickers:
                metadata = fetch_stock_metadata(ticker)
                if metadata:
                    try:
                        conn.execute(
                            text(
                                """
                                INSERT INTO stock_info (ticker, company_name, sector, market_cap, pe_ratio, eps, earnings_date)
                                VALUES (:ticker, :company_name, :sector, :market_cap, :pe_ratio, :eps, :earnings_date)
                                ON CONFLICT (ticker) DO UPDATE
                                SET company_name = EXCLUDED.company_name,
                                    sector = EXCLUDED.sector,
                                    market_cap = EXCLUDED.market_cap,
                                    pe_ratio = EXCLUDED.pe_ratio,
                                    eps = EXCLUDED.eps,
                                    earnings_date = EXCLUDED.earnings_date;
                                """
                            ),
                            metadata | {"ticker": ticker},
                        )
                        logger.info(f"Updated stock metadata for {ticker}")
                    except Exception as e:
                        logger.error(
                            f"Failed to update stock metadata for {ticker}: {e}"
                        )
                progress.update(task, advance=1)

    console.print("[bold green]✅ Stock metadata updated successfully![/bold green]")


if __name__ == "__main__":
    store_stock_metadata()
