import argparse
import logging
import os

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sqlalchemy import text

from src.tools.utils import get_database_engine, handle_exceptions, setup_logging

# ✅ Setup Logging
LOG_FILE = "data/logs/data_preprocessing.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)

# ✅ Setup Rich Console
console = Console()
logger.info("🚀 Logging setup complete for data preprocessing.")

# ✅ Ensure processed data directory exists
PROCESSED_DATA_DIR = "data/pre_processed/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


class DataPreprocessor:
    """
    Extracts, cleans, and preprocesses stock, sentiment, and technical indicator data
    from PostgreSQL for machine learning models.
    """

    def __init__(self, tickers=None):
        """Initialize database connection and scalers."""
        self.engine = get_database_engine()
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.tickers = tickers if tickers else self.fetch_tickers()

    @handle_exceptions
    def fetch_tickers(self):
        """Fetches distinct tickers from the stocks table."""
        query = "SELECT DISTINCT ticker FROM stocks;"
        with self.engine.connect() as conn:
            tickers = [row[0] for row in conn.execute(text(query))]
        console.print(
            Panel(
                f"✅ Found [cyan]{len(tickers)}[/cyan] unique tickers.", style="green"
            )
        )
        return tickers

    @handle_exceptions
    def fetch_data(self, table_name, ticker):
        """Fetches data for a specific ticker while excluding `id` columns."""
        query = f"SELECT * FROM {table_name} WHERE ticker = '{ticker}'"
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df is not None and not df.empty:
            logger.info(f"✅ Loaded {len(df)} records from {table_name} for {ticker}")

            # ✅ Drop `id` columns
            id_columns = [col for col in df.columns if "id" in col.lower()]
            df.drop(columns=id_columns, errors="ignore", inplace=True)

            # ✅ Standardize Date Column for Merging
            if table_name == "news_sentiment":
                df = df[
                    ["ticker", "published_at", "sentiment_score"]
                ]  # ✅ Select only necessary columns
                df.rename(columns={"published_at": "date"}, inplace=True)
                df["date"] = pd.to_datetime(
                    df["date"]
                ).dt.date  # ✅ Strip time component
                df["date"] = pd.to_datetime(
                    df["date"]
                )  # ✅ Convert back to datetime for merging

            else:
                df["date"] = pd.to_datetime(
                    df["date"], errors="coerce"
                )  # ✅ Convert all date columns

            return df
        else:
            logger.warning(f"⚠ No data found in {table_name} for {ticker}")
            return None

    @handle_exceptions
    def merge_data(self, stocks, indicators, sentiment):
        """Merges OHLCV, technical indicators, and sentiment data."""
        if stocks is None or indicators is None:
            logger.warning("⚠ Skipping merge due to missing stock/indicator data.")
            return None

        # ✅ Ensure all date columns are `datetime64[ns]`
        stocks["date"] = pd.to_datetime(stocks["date"])
        indicators["date"] = pd.to_datetime(indicators["date"])
        if sentiment is not None:
            sentiment["date"] = pd.to_datetime(sentiment["date"])

            # ✅ Step 1: Backward Fill for Short Gaps (Up to 2 Days)
            sentiment["sentiment_score"] = sentiment.groupby("ticker")[
                "sentiment_score"
            ].fillna(method="bfill", limit=2)

            # ✅ Step 2: Apply Rolling Mean (7-day) If Sufficient Data Exists
            sentiment["sentiment_score"] = sentiment.groupby("ticker")[
                "sentiment_score"
            ].transform(
                lambda x: x.rolling(7, min_periods=1).mean() if x.count() > 10 else x
            )

        df = stocks.merge(indicators, on=["ticker", "date"], how="left")

        if sentiment is not None:
            df = df.merge(sentiment, on=["ticker", "date"], how="left")

        # ✅ Drop any remaining `id_x` or `id_y` columns after merging
        id_columns = [col for col in df.columns if "id_" in col.lower()]
        df.drop(columns=id_columns, errors="ignore", inplace=True)

        logger.info(f"✅ Merged dataset created with {len(df)} records.")
        return df

    @handle_exceptions
    def feature_engineering(self, df):
        """Creates new lag and rolling window features."""
        if df is None:
            logger.error("❌ Feature engineering skipped due to missing data.")
            return None

        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=5).std()

        for lag in [1, 5, 10]:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

        df.fillna(0, inplace=True)
        logger.info("✅ Feature engineering completed successfully.")
        return df

    @handle_exceptions
    def normalize_features(self, df):
        """Normalizes numerical features using StandardScaler."""
        if df is None:
            logger.error("❌ Normalization skipped due to missing data.")
            return None

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            logger.warning("⚠ No numeric columns found for normalization.")
            return df

        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        logger.info("✅ Feature normalization completed successfully.")
        return df

    @handle_exceptions
    def save_to_postgres(self, df, ticker):
        """Stores processed data in PostgreSQL and updates existing records on conflict."""
        if df is None or df.empty:
            logger.error(f"❌ Skipping database save for {ticker} due to missing data.")
            return

        # ✅ Define PostgreSQL upsert query (ON CONFLICT DO UPDATE)
        sql = text(
            """
            INSERT INTO processed_stock_data (ticker, date, open, high, low, close, volume, adjusted_close,
                                            sma_50, sma_200, ema_50, ema_200, rsi_14, adx_14, atr_14, cci_20,
                                            williamsr_14, macd, macd_signal, macd_hist, bb_upper, bb_lower,
                                            stoch_k, stoch_d, sentiment_score, returns, volatility,
                                            close_lag_1, volume_lag_1, close_lag_5, volume_lag_5,
                                            close_lag_10, volume_lag_10)
            VALUES (:ticker, :date, :open, :high, :low, :close, :volume, :adjusted_close,
                    :sma_50, :sma_200, :ema_50, :ema_200, :rsi_14, :adx_14, :atr_14, :cci_20,
                    :williamsr_14, :macd, :macd_signal, :macd_hist, :bb_upper, :bb_lower,
                    :stoch_k, :stoch_d, :sentiment_score, :returns, :volatility,
                    :close_lag_1, :volume_lag_1, :close_lag_5, :volume_lag_5,
                    :close_lag_10, :volume_lag_10)
            ON CONFLICT (ticker, date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                adjusted_close = EXCLUDED.adjusted_close,
                sma_50 = EXCLUDED.sma_50,
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
                bb_lower = EXCLUDED.bb_lower,
                stoch_k = EXCLUDED.stoch_k,
                stoch_d = EXCLUDED.stoch_d,
                sentiment_score = EXCLUDED.sentiment_score,
                returns = EXCLUDED.returns,
                volatility = EXCLUDED.volatility,
                close_lag_1 = EXCLUDED.close_lag_1,
                volume_lag_1 = EXCLUDED.volume_lag_1,
                close_lag_5 = EXCLUDED.close_lag_5,
                volume_lag_5 = EXCLUDED.volume_lag_5,
                close_lag_10 = EXCLUDED.close_lag_10,
                volume_lag_10 = EXCLUDED.volume_lag_10;
        """
        )

        # ✅ Convert DataFrame to Dictionary for Bulk Insert
        data_dicts = df.to_dict(orient="records")

        with self.engine.begin() as conn:
            conn.execute(sql, data_dicts)

        logger.info(f"✅ Processed data for {ticker} saved and updated in PostgreSQL.")

    @handle_exceptions
    def save_to_csv(self, df, ticker):
        """Saves processed data to a CSV file."""
        if df is None or df.empty:
            logger.error(f"❌ Skipping CSV save for {ticker} due to missing data.")
            return

        file_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"✅ Processed data for {ticker} saved to {file_path}")

    @handle_exceptions
    def process(self):
        """Executes the full data preprocessing pipeline for selected tickers."""
        console.print(
            Panel(
                f"🚀 [bold cyan]Processing {len(self.tickers)} tickers...[/bold cyan]",
                style="cyan",
            )
        )
        logger.info(f"🚀 Processing {len(self.tickers)} tickers.")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue] Processing {task.fields[ticker]}..."),
            console=console,
        ) as progress:
            task = progress.add_task("", total=len(self.tickers), ticker="Starting...")

            for ticker in self.tickers:
                progress.update(task, ticker=ticker)

                stocks = self.fetch_data("stocks", ticker)
                indicators = self.fetch_data("technical_indicators", ticker)
                sentiment = self.fetch_data("news_sentiment", ticker)

                if stocks is None or indicators is None:
                    logger.error(f"❌ Data fetching failed for {ticker}. Skipping.")
                    progress.advance(task)
                    continue

                df = self.merge_data(stocks, indicators, sentiment)
                df = self.feature_engineering(df)
                df = self.normalize_features(df)

                self.save_to_postgres(df, ticker)
                self.save_to_csv(df, ticker)

                progress.advance(task)

        console.print(
            Panel(
                "✅ [bold green]All requested tickers processed successfully![/bold green]",
                style="green",
            )
        )
        logger.info("✅ All requested tickers processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess stock data for machine learning."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Comma-separated list of tickers to process. Defaults to all tickers.",
    )
    args = parser.parse_args()
    tickers = args.ticker.split(",") if args.ticker else None

    processor = DataPreprocessor(tickers=tickers)
    processor.process()
