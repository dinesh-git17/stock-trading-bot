import argparse
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from src.tools.utils import get_database_engine, handle_exceptions, setup_logging

# ✅ Setup Logging
LOG_FILE = "data/logs/xgboost_model.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)

# ✅ Setup Rich Console
console = Console()
logger.info("🚀 Logging setup complete for XGBoost model training.")

# ✅ Ensure model directory exists
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)


class XGBoostStockPredictor:
    """
    Trains an optimized XGBoost model for stock price prediction.
    """

    def __init__(self, tickers=None, plot=False, random_state=42):
        self.tickers = tickers if tickers else self.fetch_tickers()
        self.plot = plot  # ✅ Store plot option
        self.random_state = random_state
        self.engine = get_database_engine()
        self.scaler = StandardScaler()

    @handle_exceptions
    def fetch_tickers(self):
        """Fetches distinct tickers from the database."""
        query = "SELECT DISTINCT ticker FROM processed_stock_data;"
        with self.engine.connect() as conn:
            tickers = [row[0] for row in conn.execute(text(query))]

        console.print(
            Panel(
                f"✅ Found [cyan]{len(tickers)}[/cyan] unique tickers.", style="green"
            )
        )
        return tickers

    @handle_exceptions
    def fetch_data(self, ticker):
        """Fetches processed stock data for a specific ticker."""
        query = f"SELECT * FROM processed_stock_data WHERE ticker = '{ticker}'"
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df is not None and not df.empty:
            logger.info(f"✅ Loaded {len(df)} records for {ticker}")

            # ✅ Drop unnecessary columns
            df.drop(columns=["ticker"], inplace=True, errors="ignore")

            # ✅ Ensure `date` is sorted
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values(by="date", inplace=True)
            df.set_index("date", inplace=True)
            return df
        else:
            logger.warning(f"⚠ No data found for {ticker}")
            return None

    @handle_exceptions
    def prepare_features(self, df):
        """Prepares features and target variable for training."""
        if df is None:
            logger.error("❌ Data preparation skipped due to missing data.")
            return None, None, None

        target = "close"
        features = [
            "open",
            "high",
            "low",
            "volume",
            "sma_50",
            "ema_50",
            "rsi_14",
            "adx_14",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_lower",
            "stoch_k",
            "stoch_d",
            "sentiment_score",
            "close_lag_1",
            "close_lag_5",
            "close_lag_10",
        ]

        # ✅ Normalize Features
        X = self.scaler.fit_transform(df[features])
        y = self.scaler.fit_transform(df[[target]]).flatten()  # ✅ Normalize `close`

        return X, y, df.index  # ✅ Return dates for plotting

    @handle_exceptions
    def train_model(self, X, y):
        """Trains an XGBoost model with optimized hyperparameters."""
        tscv = TimeSeriesSplit(n_splits=5)

        model = xgb.XGBRegressor(
            objective="reg:squarederror", random_state=self.random_state, n_jobs=-1
        )

        param_grid = {
            "n_estimators": [100, 300],  # Only test 2 values instead of 3
            "learning_rate": [0.01, 0.05],  # Reduce learning rate options
            "max_depth": [4, 6],  # Only test 2 values instead of 3
            "subsample": [0.8],  # Fixed to 1 value
            "colsample_bytree": [0.8],  # Fixed to 1 value
        }

        grid_search = GridSearchCV(
            model,
            param_grid,
            scoring="neg_mean_squared_error",
            cv=tscv,
            verbose=1,
            n_jobs=-1,
        )
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        logger.info(f"✅ Best model parameters: {grid_search.best_params_}")
        return best_model

    @handle_exceptions
    def evaluate_model(self, model, X, y, dates, ticker):
        """Evaluates model performance and logs results."""
        predictions = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)

        logger.info(f"📊 Model Performance for {ticker}:")
        logger.info(f"   - RMSE: {rmse:.4f}")
        logger.info(f"   - R² Score: {r2:.4f}")

        # ✅ Show plot if `--plot` flag is used
        if self.plot:
            self.plot_predictions(dates, y, predictions, ticker)

        return rmse, r2

    def plot_predictions(self, dates, actual, predicted, ticker):
        """Plots Actual vs. Predicted stock prices."""
        plt.figure(figsize=(10, 5))
        plt.plot(dates, actual, label="Actual", color="blue")
        plt.plot(dates, predicted, label="Predicted", color="red", linestyle="dashed")
        plt.title(f"{ticker} - Actual vs. Predicted Prices")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid()
        plt.show()

    @handle_exceptions
    def save_model(self, model, ticker):
        """Saves trained XGBoost model."""
        model_path = os.path.join(MODEL_DIR, f"xgboost_{ticker}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"✅ Model saved to {model_path}")

    @handle_exceptions
    def run_pipeline(self):
        """Executes the full training pipeline."""
        console.print(
            Panel(
                f"🚀 [bold cyan]Training XGBoost Model for {len(self.tickers)} tickers...[/bold cyan]",
                style="cyan",
            )
        )
        logger.info(f"🚀 Training XGBoost Model for {len(self.tickers)} tickers")

        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue] Training {task.fields[ticker]}..."),
            console=console,
        ) as progress:
            task = progress.add_task("", total=len(self.tickers), ticker="Starting...")

            for ticker in self.tickers:
                progress.update(task, ticker=ticker)

                df = self.fetch_data(ticker)
                X, y, dates = self.prepare_features(df)

                if X is None or y is None:
                    logger.error(f"❌ Skipping {ticker} due to missing data.")
                    progress.advance(task)
                    continue

                model = self.train_model(X, y)
                rmse, r2 = self.evaluate_model(model, X, y, dates, ticker)
                self.save_model(model, ticker)

                results.append((ticker, rmse, r2))
                progress.advance(task)

        table = Table(title="XGBoost Model Performance", show_lines=True)
        table.add_column("Ticker", style="cyan", justify="center")
        table.add_column("RMSE", style="green", justify="center")
        table.add_column("R² Score", style="green", justify="center")

        for ticker, rmse, r2 in results:
            table.add_row(ticker, f"{rmse:.4f}", f"{r2:.4f}")

        console.print(table)
        console.print(
            Panel(
                "✅ [bold green]All models trained successfully![/bold green]",
                style="green",
            )
        )
        logger.info("✅ All models trained successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for stock price prediction."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Comma-separated list of tickers. Defaults to all tickers.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a plot of actual vs. predicted values.",
    )
    args = parser.parse_args()

    tickers = args.ticker.split(",") if args.ticker else None
    predictor = XGBoostStockPredictor(tickers=tickers, plot=args.plot)
    predictor.run_pipeline()
