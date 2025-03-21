import argparse
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import openai
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from sqlalchemy import text

from src.tools.utils import (
    get_database_engine,
    handle_exceptions,
    setup_logging,
)

# ✅ Logging Setup
LOG_FILE = "data/logs/stockgpt_embeddings.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)

# ✅ Console Setup
console = Console()

# ✅ OpenAI Setup
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# ✅ Embedding Save Directory
EMBED_SAVE_DIR = "data/embeddings/"
os.makedirs(EMBED_SAVE_DIR, exist_ok=True)

# ✅ Persistent Embedding Cache Path
CACHE_PATH = os.path.join(EMBED_SAVE_DIR, "embedding_cache.pkl")


class StockGPTEngine:
    """
    Embeds financial news using OpenAI and stores daily average embeddings.
    """

    def __init__(self):
        self.engine = get_database_engine()
        self.embedding_cache = self.load_embedding_cache()

    def load_embedding_cache(self):
        """Load saved embedding cache from disk if available."""
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
                console.print(
                    f"[green]✅ Loaded {len(cache)} cached embeddings from disk.[/green]"
                )
                return cache
        return {}

    def save_embedding_cache(self):
        """Persist the embedding cache to disk."""
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(self.embedding_cache, f)
        console.print(f"[green]💾 Saved embedding cache to {CACHE_PATH}[/green]")

    @handle_exceptions
    def fetch_all_tickers(self):
        """Fetch distinct tickers from the news_sentiment table."""
        query = "SELECT DISTINCT ticker FROM news_sentiment ORDER BY ticker;"
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            tickers = [row[0] for row in result]
        return tickers

    @handle_exceptions
    def fetch_news_for_ticker(self, ticker):
        """Fetch all news for a ticker."""
        query = """
        SELECT title, description, published_at
        FROM news_sentiment
        WHERE ticker = :ticker
        ORDER BY published_at;
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"ticker": ticker})
        return df

    @staticmethod
    def format_text(title, description):
        """Format title and description into a single string."""
        if not title and not description:
            return None
        return f"{title}. {description}" if description else title

    @handle_exceptions
    def embed_text(self, text):
        """Get OpenAI embedding for a single text string with caching."""
        if not text or len(text.strip()) == 0:
            return None

        if text in self.embedding_cache:
            return self.embedding_cache[text]

        response = openai.embeddings.create(input=[text], model=EMBED_MODEL)
        embedding = np.array(response.data[0].embedding)
        self.embedding_cache[text] = embedding
        return embedding

    def group_and_embed(self, df):
        """Group news by day and average embeddings."""
        grouped = defaultdict(list)

        for _, row in df.iterrows():
            published_date = pd.to_datetime(row["published_at"]).date()
            text = self.format_text(row["title"], row["description"])

            try:
                emb = self.embed_text(text)
                if emb is not None:
                    grouped[published_date].append(emb)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"❌ Failed to embed article on {published_date}: {e}")
                continue

        daily_embeddings = {
            date: np.mean(embeds, axis=0)
            for date, embeds in grouped.items()
            if len(embeds) > 0
        }

        return daily_embeddings

    def save_embeddings(self, ticker, embedding_dict):
        """Save ticker embedding dict to .npz file."""
        output_path = os.path.join(EMBED_SAVE_DIR, f"{ticker}.npz")
        np.savez(
            output_path,
            dates=list(embedding_dict.keys()),
            embeddings=list(embedding_dict.values()),
        )
        console.print(f"[green]✅ Saved embeddings to {output_path}[/green]")

    @handle_exceptions
    def run(self, tickers=None):
        """Main execution pipeline."""
        if tickers is None:
            tickers = self.fetch_all_tickers()
        if not tickers:
            console.print("[red]❌ No tickers found to process.[/red]")
            return

        console.print(
            Panel(
                f"🧠 [bold cyan]Generating StockGPT Embeddings for {len(tickers)} tickers...[/bold cyan]",
                style="cyan",
            )
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Embedding {task.fields[ticker]}..."),
            console=console,
        ) as progress:
            task = progress.add_task("", total=len(tickers), ticker="")

            for ticker in tickers:
                progress.update(task, ticker=ticker)

                df = self.fetch_news_for_ticker(ticker)
                if df is None or df.empty:
                    console.print(f"[yellow]⚠ No news found for {ticker}[/yellow]")
                    progress.advance(task)
                    continue

                emb_dict = self.group_and_embed(df)
                if not emb_dict:
                    console.print(f"[red]❌ No embeddings generated for {ticker}[/red]")
                    progress.advance(task)
                    continue

                self.save_embeddings(ticker, emb_dict)
                progress.advance(task)

        self.save_embedding_cache()

        console.print(
            Panel(
                "[bold green]🎉 All ticker embeddings generated and saved successfully![/bold green]",
                style="green",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate StockGPT embeddings for selected tickers."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Comma-separated list of tickers to process. If not provided, all tickers will be used.",
    )
    args = parser.parse_args()

    tickers = args.ticker.split(",") if args.ticker else None
    StockGPTEngine().run(tickers=tickers)
