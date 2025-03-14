import logging
import os

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.theme import Theme
from sqlalchemy import text

from src.ml.data_preprocessing import preprocess_data
from src.tools.utils import get_database_engine, setup_logging

# ✅ Setup Logging
setup_logging("data/logs/fetch_training_data.log")

# ✅ Custom Theme for Console
custom_theme = Theme(
    {
        "success": "bold green",
        "info": "bold blue",
        "warning": "bold yellow",
        "error": "bold red",
    }
)
console = Console(theme=custom_theme)

logging.info("🚀 Starting fetch_training_data script...")


def fetch_training_data(chunksize: int = 5000) -> pd.DataFrame:
    """
    Fetches stock market data required for ML model training and passes it to preprocessing.

    Args:
        chunksize (int): Number of rows per chunk.

    Returns:
        pd.DataFrame: Preprocessed dataset ready for ML models.
    """
    logging.info("📡 Connecting to the database...")
    engine = get_database_engine()

    console.print("[info]📥 Fetching training data from database...[/info]")
    logging.info("📥 Fetching training data from database...")

    query = """
    SELECT s.ticker, s.date, s.open, s.high, s.low, s.close, s.volume,
           ti.sma_50, ti.sma_200, ti.ema_50, ti.ema_200, 
           ti.macd, ti.macd_signal, ti.macd_hist,
           ti.bb_upper, ti.bb_middle, ti.bb_lower,
           ti.rsi_14, ti.adx_14, ti.atr_14, ti.cci_20, ti.williamsr_14, 
           ti.stoch_k, ti.stoch_d,
           ns.sentiment_score
    FROM stocks s
    LEFT JOIN technical_indicators ti 
        ON s.ticker = ti.ticker AND s.date = ti.date
    LEFT JOIN news_sentiment ns 
        ON s.ticker = ns.ticker AND ns.published_at::DATE = s.date::DATE
    ORDER BY s.date;
    """

    with engine.connect() as conn, Progress(
        TextColumn("[info]⏳ Downloading:[/]"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching data...", total=1)

        data_chunks = []
        logging.info("🛠️ Querying database and fetching data in chunks...")
        for chunk in pd.read_sql_query(text(query), conn, chunksize=chunksize):
            data_chunks.append(chunk)
            progress.update(task, advance=1)

    final_data = pd.concat(data_chunks, ignore_index=True)
    logging.info(f"✅ Data fetching complete. Total rows fetched: {len(final_data)}")

    console.print("[success]✅ Fetching complete! Now preprocessing data...[/success]")
    logging.info("📊 Sending data for preprocessing...")

    # ✅ Send data directly to preprocessing
    preprocess_data(final_data)

    logging.info("🏁 Fetching & Preprocessing Complete!")


if __name__ == "__main__":
    fetch_training_data()
