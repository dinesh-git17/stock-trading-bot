import logging
import os
import pickle

import pandas as pd
import sqlalchemy
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/data_preprocessing.log"
os.makedirs("data/logs", exist_ok=True)

# ✅ Insert blank lines before logging new logs
with open(LOG_FILE, "a") as log_file:
    log_file.write("\n" * 3)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ Database Connection Settings (from .env)
DB_URI = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URI)

# ✅ Directories
OUTPUT_DIR = "data/transformed"
SENTIMENT_DIR = "data/sentiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SENTIMENT_DIR, exist_ok=True)


# ✅ Function to Fetch Sentiment Data
def fetch_sentiment_data(ticker):
    """Fetch sentiment scores from PostgreSQL and save them for the given ticker."""
    logging.info(f"🔍 Fetching sentiment data for {ticker}...")
    console.print(f"[cyan]🔍 Fetching sentiment data for {ticker}...[/cyan]")

    try:
        query = """
            SELECT published_at AS date, sentiment_score
            FROM news_sentiment
            WHERE ticker = %s
            ORDER BY published_at ASC;
        """
        df = pd.read_sql(query, engine, params=(ticker,))  # ✅ Pass tuple (ticker,)

        if df.empty:
            logging.warning(f"⚠️ No sentiment data found for {ticker}. Skipping...")
            console.print(
                f"[yellow]⚠️ No sentiment data found for {ticker}. Skipping...[/yellow]"
            )
            return None

        # ✅ Align sentiment scores by shifting to match stock prices
        df["sentiment_score"] = df["sentiment_score"].shift(7)
        df["sentiment_score"] = df["sentiment_score"].bfill()  # ✅ Fixes FutureWarning

        # ✅ Save sentiment data
        sentiment_path = f"{SENTIMENT_DIR}/sentiment_{ticker}.pkl"
        with open(sentiment_path, "wb") as f:
            pickle.dump(df, f)

        logging.info(f"✅ Sentiment data for {ticker} saved at {sentiment_path}")
        console.print(f"[green]✅ Sentiment data for {ticker} saved.[/green]")

    except Exception as e:
        logging.error(f"❌ Error fetching sentiment data for {ticker}: {e}")
        console.print(f"[red]❌ Error fetching sentiment data for {ticker}: {e}[/red]")


# ✅ Function to Fetch and Store Sentiment for All Tickers
def process_all_tickers():
    """Fetch sentiment data for all tickers in the database."""
    logging.info("🚀 Fetching sentiment data for all tickers...")
    console.print(
        "[bold blue]🚀 Fetching sentiment data for all tickers...[/bold blue]"
    )

    try:
        query = "SELECT DISTINCT ticker FROM news_sentiment;"
        tickers = pd.read_sql(query, engine)["ticker"].tolist()
    except Exception as e:
        logging.error(f"❌ Failed to fetch tickers: {e}")
        console.print(f"[red]❌ Failed to fetch tickers: {e}[/red]")
        return

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Processing Sentiment Data...", total=len(tickers))
        for ticker in tickers:
            fetch_sentiment_data(ticker)
            progress.update(task, advance=1)

    logging.info("✅ Sentiment data fetching complete!")
    console.print("[bold green]✅ Sentiment data fetching complete![/bold green]")


if __name__ == "__main__":
    process_all_tickers()
