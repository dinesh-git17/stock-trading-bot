import os

# ✅ Suppress TensorFlow & Metal Plugin logs (macOS/M1)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import argparse
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from tensorflow.keras import callbacks, layers, models, regularizers

from src.tools.utils import (
    get_database_engine,
    handle_exceptions,
    setup_logging,
)

# ✅ Suppress TensorFlow logs
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ✅ Logging Setup
LOG_FILE = "data/logs/stockgpt_model.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)

# ✅ Console Setup
console = Console()

# ✅ Constants
MODEL_DIR = "models/"
EMBED_DIR = "data/embeddings/"
os.makedirs(MODEL_DIR, exist_ok=True)
LOOKBACK_DAYS = 10  # 🔥 Longer memory
PCA_COMPONENTS = 8  # 🔥 Reduce embeddings to key features

# ✅ Optimized Feature Set
NUMERIC_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
    "returns",
    "volatility",
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
    "bb_lower",
    "stoch_k",
    "stoch_d",
    "sentiment_score",
    "close_lag_1",
    "close_lag_5",
    "close_lag_10",
    "volume_lag_1",
    "volume_lag_5",
    "volume_lag_10",
    "bollinger_upper",
    "bollinger_lower",
    "momentum_10",
]


class StockGPTModel:
    def __init__(self, ticker, plot=False):
        self.ticker = ticker
        self.plot = plot
        self.engine = get_database_engine()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=PCA_COMPONENTS)

    @handle_exceptions
    def load_price_data(self):
        """Fetch stock price data from DB."""
        query = f"SELECT * FROM processed_stock_data WHERE ticker = '{self.ticker}';"
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.set_index("date", inplace=True)
        return df

    @handle_exceptions
    def load_embeddings(self):
        """Load precomputed news embeddings."""
        path = os.path.join(EMBED_DIR, f"{self.ticker}.npz")
        if not os.path.exists(path):
            console.print(f"[red]❌ No embedding file found for {self.ticker}[/red]")
            return None
        data = np.load(path, allow_pickle=True)
        dates = pd.to_datetime(data["dates"])
        embeddings = np.stack(data["embeddings"])
        return pd.DataFrame({"date": dates, "embedding": list(embeddings)})

    @handle_exceptions
    def prepare_data(self, price_df, embed_df):
        """Prepares dataset for model training, adding missing indicators."""
        df = price_df.copy()

        # ✅ Ensure embeddings exist
        if embed_df is None or embed_df.empty:
            logger.warning(f"⚠ No embeddings available for {self.ticker}. Skipping.")
            return None, None, None

        embed_df.set_index("date", inplace=True)
        df = df.join(embed_df, how="left")  # 🔥 Use "left" join instead of "inner"

        # ✅ Fill missing embeddings with the previous day's
        df["embedding"] = df["embedding"].fillna(method="ffill")

        # ✅ Compute missing indicators
        df["bollinger_upper"] = df["sma_50"] + (df["close"].rolling(20).std() * 2)
        df["bollinger_lower"] = df["sma_50"] - (df["close"].rolling(20).std() * 2)
        df["momentum_10"] = df["close"].diff(10)
        df["volatility_10"] = df["close"].pct_change().rolling(10).std()
        df["rsi_7"] = df["rsi_14"].rolling(7).mean()

        # ✅ Compute `relative_close`
        df["relative_close"] = (df["close"] - df["close_lag_1"]) / df["close_lag_1"]

        # ✅ Drop rows with NaNs after feature engineering
        df.dropna(inplace=True)

        if df.empty:
            logger.error(
                f"❌ No data left after feature computation for {self.ticker}."
            )
            return None, None, None

        # ✅ PCA on Embeddings
        embed_matrix = np.stack(df["embedding"].dropna().values)
        reduced_embeddings = self.pca.fit_transform(embed_matrix)
        df.drop(columns=["embedding"], inplace=True)
        df = df.join(pd.DataFrame(reduced_embeddings, index=df.index))

        # ✅ Scale Features
        X_numeric_scaled = self.scaler.fit_transform(df[NUMERIC_FEATURES])
        X = np.hstack([X_numeric_scaled, reduced_embeddings])

        # ✅ Target (relative close change)
        y = df["relative_close"].shift(-LOOKBACK_DAYS).iloc[:-LOOKBACK_DAYS].values
        X = X[:-LOOKBACK_DAYS]

        # ✅ Dates for plotting
        dates = df.index[:-LOOKBACK_DAYS]

        # ✅ Ensure we have data
        if X.shape[0] == 0:
            logger.error(f"❌ Not enough processed data for {self.ticker}.")
            return None, None, None

        return X, y, dates

    def train_with_mlp(self, X, y, dates):
        """Fallback model: simple feedforward MLP for low-data tickers."""
        console.print(
            Panel(
                "[yellow]⚠️ Low data: Using MLP Fallback Model[/yellow]", style="yellow"
            )
        )

        model = models.Sequential(
            [
                layers.Input(shape=(X.shape[1],)),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(64, activation="relu"),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        history = model.fit(
            X,
            y,
            epochs=50,
            batch_size=16,
            validation_split=0.1,
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )

        self.display_training_summary(history, X, y, model)

        if self.plot:
            self.plot_predictions(model, X, y, dates)

        path = os.path.join(MODEL_DIR, f"stockgpt_{self.ticker}_mlp.keras")
        model.save(path)
        console.print(f"[green]✅ MLP model saved to {path}[/green]")
        return model

    @handle_exceptions
    def build_model(self, input_dim):
        """Constructs a hybrid LSTM + Dense model."""
        inputs = layers.Input(shape=(LOOKBACK_DAYS, input_dim))
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.LSTM(64, return_sequences=False)(x)
        x = layers.Dense(
            32, activation="relu", kernel_regularizer=regularizers.l2(0.001)
        )(x)
        x = layers.LayerNormalization()(x)  # 🔥 Normalize activations
        outputs = layers.Dense(1, activation="linear")(x)  # 🔥 Predict % Change

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    @handle_exceptions
    def train(self, X, y, dates):
        """Trains LSTM if enough data, otherwise MLP fallback."""
        if X.shape[0] < LOOKBACK_DAYS * 2:
            return self.train_with_mlp(X, y, dates)

        # ✅ Trim to multiple of LOOKBACK_DAYS
        num_samples = X.shape[0] - (X.shape[0] % LOOKBACK_DAYS)
        if num_samples < LOOKBACK_DAYS * 2:
            return self.train_with_mlp(X, y, dates)

        X, y, dates = X[:num_samples], y[:num_samples], dates[:num_samples]

        # ✅ Reshape for LSTM
        X = X.reshape((X.shape[0] // LOOKBACK_DAYS, LOOKBACK_DAYS, -1))

        if X.shape[0] < 2:
            return self.train_with_mlp(X.reshape((X.shape[0], -1)), y, dates)

        model = self.build_model(X.shape[2])

        history = model.fit(
            X,
            y,
            epochs=100,
            batch_size=64,
            validation_split=0.1,
            callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0,
        )

        self.display_training_summary(history, X, y, model)

        if self.plot:
            self.plot_predictions(model, X, y, dates)

        path = os.path.join(MODEL_DIR, f"stockgpt_{self.ticker}.keras")
        model.save(path)
        console.print(f"[green]✅ LSTM model saved to {path}[/green]")
        return model

    def display_training_summary(self, history, X, y, model):
        """Logs training results."""
        y_pred = model.predict(X).flatten()
        r2 = r2_score(y, y_pred)

        table = Table(
            title=f"[bold cyan]Training Summary for {self.ticker}[/bold cyan]"
        )
        table.add_column("Metric", style="bold yellow", justify="center")
        table.add_column("Value", style="bold green", justify="center")
        table.add_row("Final Loss (MSE)", f"{history.history['loss'][-1]:.4f}")
        table.add_row("Final MAE", f"{history.history['mae'][-1]:.4f}")
        table.add_row("R² Score", f"{r2:.4f}")
        console.print(table)

    def plot_predictions(self, model, X, y, dates):
        y_pred = model.predict(X).flatten()
        plt.figure(figsize=(12, 5))
        plt.plot(dates, y, label="Actual", color="blue")
        plt.plot(dates, y_pred, label="Predicted", color="red", linestyle="--")
        plt.title(f"{self.ticker} - Actual vs Predicted Close Price ($)")
        plt.xlabel("Date")
        plt.ylabel("Close Price ($)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    @handle_exceptions
    def run(self):
        console.print(
            Panel(
                f"🚀 [bold cyan]Training StockGPT for {self.ticker}[/bold cyan]",
                style="cyan",
            )
        )

        price_df = self.load_price_data()
        embed_df = self.load_embeddings()

        if price_df is None or embed_df is None:
            return

        X, y, dates = self.prepare_data(price_df, embed_df)
        if X is None or y is None:
            return

        self.train(X, y, dates)

        console.print(
            Panel(
                f"✅ [bold green]Training complete for {self.ticker}[/bold green]",
                style="green",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train StockGPT on historical news + prices."
    )
    parser.add_argument(
        "--ticker", type=str, required=True, help="Ticker symbol (e.g., TSLA)"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot actual vs predicted prices"
    )
    args = parser.parse_args()

    StockGPTModel(ticker=args.ticker, plot=args.plot).run()
