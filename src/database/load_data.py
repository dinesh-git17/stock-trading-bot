import logging
import os

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from sqlalchemy import text

from src.tools.utils import get_database_engine, handle_exceptions, setup_logging

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/load_data.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)  # Ensure log directory exists
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)
console = Console()

# ✅ Data Directories
CLEANED_DATA_DIR = "data/cleaned/"


@handle_exceptions
def insert_data(engine, ticker, df):
    """
    Inserts cleaned stock data into `stocks` and `technical_indicators` tables using batch processing.
    Ensures all required indicator columns exist and fills missing ones with None.
    """
    try:
        stock_columns = [
            "ticker",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adjusted_close",
        ]
        indicator_columns = [
            "ticker",
            "date",
            "sma_50",
            "sma_200",
            "ema_50",
            "ema_200",
            "rsi_14",
            "adx_14",
            "atr_14",
            "cci_20",
            "williamsr_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "stoch_k",
            "stoch_d",
        ]

        df["ticker"] = ticker  # ✅ Ensure ticker column exists
        df.rename(columns={"Date": "date"}, inplace=True)  # ✅ Standardize column names

        # ✅ Ensure missing indicator columns are filled with None
        for col in indicator_columns:
            if col not in df.columns:
                df[col] = None  # Fill missing columns

        stock_values = df[stock_columns].to_dict(orient="records")
        indicator_values = df[indicator_columns].to_dict(orient="records")

        stock_query = text(
            f"""
            INSERT INTO stocks ({", ".join(stock_columns)}) 
            VALUES ({", ".join([":%s" % col for col in stock_columns])})
            ON CONFLICT (ticker, date) DO UPDATE 
            SET open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                adjusted_close = EXCLUDED.adjusted_close;
        """
        )

        indicator_query = text(
            f"""
            INSERT INTO technical_indicators ({", ".join(indicator_columns)}) 
            VALUES ({", ".join([":%s" % col for col in indicator_columns])})
            ON CONFLICT (ticker, date) DO UPDATE 
            SET sma_50 = EXCLUDED.sma_50,
                sma_200 = EXCLUDED.sma_200,
                ema_50 = EXCLUDED.ema_50,
                ema_200 = EXCLUDED.ema_200,
                rsi_14 = EXCLUDED.rsi_14,
                adx_14 = EXCLUDED.adx_14,
                atr_14 = EXCLUDED.atr_14,
                cci_20 = EXCLUDED.cci_20,
                williamsr_14 = EXCLUDED.williamsr_14,
                macd = EXCLUDED.macd,
                macd_signal = EXCLUDED.macd_signal,
                macd_hist = EXCLUDED.macd_hist,
                bb_upper = EXCLUDED.bb_upper,
                bb_middle = EXCLUDED.bb_middle,
                bb_lower = EXCLUDED.bb_lower,
                stoch_k = EXCLUDED.stoch_k,
                stoch_d = EXCLUDED.stoch_d;
        """
        )

        with engine.begin() as connection:  # ✅ Ensure the transaction commits properly
            try:
                connection.execute(stock_query, stock_values)
            except Exception as e:
                logger.error(
                    f"❌ Error inserting stock data for {ticker}: {e}", exc_info=True
                )

            try:
                connection.execute(indicator_query, indicator_values)
            except Exception as e:
                logger.error(
                    f"❌ Error inserting indicator data for {ticker}: {e}",
                    exc_info=True,
                )

        logger.info(f"✅ Successfully loaded data for {ticker}")

    except Exception as e:
        console.print(f"[bold red]❌ Error inserting data for {ticker}:[/bold red] {e}")
        logger.error(f"Error inserting data for {ticker}: {e}", exc_info=True)


@handle_exceptions
def process_stock(engine, ticker):
    """
    Loads cleaned stock data for a given ticker and inserts it into the database.
    """
    cleaned_file = os.path.join(CLEANED_DATA_DIR, f"{ticker}_cleaned.csv")

    if not os.path.exists(cleaned_file):
        logger.warning(f"⚠ Cleaned data file missing for {ticker}")
        return

    df = pd.read_csv(cleaned_file, parse_dates=["Date"])

    if df.empty:
        logger.warning(f"⚠ No data found in {cleaned_file}")
        return

    insert_data(engine, ticker, df)


@handle_exceptions
def process_all_stocks():
    """
    Loads all cleaned stock data into the database.
    """
    engine = get_database_engine()
    if not engine:
        return

    files = [f for f in os.listdir(CLEANED_DATA_DIR) if f.endswith("_cleaned.csv")]
    tickers = [file.replace("_cleaned.csv", "") for file in files]

    console.print(
        f"\n🚀 [bold cyan]Loading cleaned stock data for {len(tickers)} stocks...[/bold cyan]\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Inserting Data into Database...[/bold yellow]"),
        console=console,
    ) as progress:
        task = progress.add_task("loading", total=len(tickers))

        for ticker in tickers:
            process_stock(engine, ticker)
            progress.update(task, advance=1)

    console.print(
        "[bold green]✅ All cleaned stock data loaded successfully![/bold green]"
    )
    logger.info("✅ All cleaned stock data loaded successfully!")


if __name__ == "__main__":
    process_all_stocks()
