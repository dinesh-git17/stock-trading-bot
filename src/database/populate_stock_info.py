import logging
import os
import time

import yfinance as yf
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sqlalchemy import text

# ✅ Import utilities
from src.tools.utils import get_database_engine, handle_exceptions, setup_logging

# ✅ Setup Logging
LOG_FILE = "data/logs/populate_stock_info.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)
console = Console()
logger.info("🚀 Logging setup complete.")


@handle_exceptions
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
            logger.warning(
                f"⚠ Retry {attempt + 1}/{max_retries} - Failed to fetch data for {ticker}: {e}"
            )
            time.sleep(2**attempt)  # Exponential backoff
    logger.error(f"❌ Max retries reached. Skipping {ticker}.")
    return {key: None for key in fetch_stock_metadata("placeholder").keys()}


@handle_exceptions
def populate_stock_info():
    """Fetches stock metadata and populates the stock_info table efficiently."""
    engine = get_database_engine()

    with engine.connect() as conn:
        tickers = [
            row[0] for row in conn.execute(text("SELECT DISTINCT ticker FROM stocks;"))
        ]

    console.print(
        f"\n🚀 Fetching and populating stock metadata for {len(tickers)} stocks...\n",
        style="bold cyan",
    )

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

        with engine.begin() as conn:
            for ticker in tickers:
                progress.update(task, ticker=ticker)
                metadata = fetch_stock_metadata(ticker)
                batch_data.append(metadata)

                if len(batch_data) >= batch_size:
                    insert_stock_info(conn, batch_data)
                    batch_data.clear()
                progress.advance(task)

            if batch_data:
                insert_stock_info(conn, batch_data)

    console.print(
        "\n[bold green]✅ Stock metadata populated successfully![/bold green]\n"
    )


@handle_exceptions
def insert_stock_info(conn, batch_data):
    """Performs batch insertion of stock metadata into PostgreSQL."""
    sql = text(
        """
        INSERT INTO stock_info (ticker, company_name, sector, industry, exchange, market_cap, pe_ratio, eps, 
                                earnings_date, ipo_date, price_to_sales_ratio, price_to_book_ratio, enterprise_value, 
                                ebitda, profit_margin, return_on_equity, beta, dividend_yield)
        VALUES (:ticker, :company_name, :sector, :industry, :exchange, :market_cap, :pe_ratio, :eps, 
                :earnings_date, :ipo_date, :price_to_sales_ratio, :price_to_book_ratio, :enterprise_value, 
                :ebitda, :profit_margin, :return_on_equity, :beta, :dividend_yield)
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
    )
    conn.execute(sql, batch_data)
    logger.info(f"✅ Batch inserted {len(batch_data)} records into stock_info.")


if __name__ == "__main__":
    populate_stock_info()
