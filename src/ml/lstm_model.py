import argparse
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from tensorflow.keras.callbacks import Callback, EarlyStopping  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    LSTM,
    Dense,
    Input,
    Layer,
    LayerNormalization,
)
from tensorflow.keras.optimizers import AdamW  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore

from src.tools.utils import get_database_engine, handle_exceptions, setup_logging

# ✅ Suppress TensorFlow Logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hide TF warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ✅ Suppress macOS system warnings (IMKClient)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ✅ Setup Logging
LOG_FILE = "data/logs/lstm_model.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)

# ✅ Setup Rich Console
console = Console()
logger.info("🚀 Logging setup complete for LSTM model training.")

# ✅ Ensure model directory exists
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)


class CyclicalLearningRate(Callback):
    """Implements cyclical learning rate (CLR) using cosine decay."""

    def __init__(self, min_lr=1e-5, max_lr=1e-3, step_size=2000):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.iterations = 0
        self.history = {}

    def on_train_batch_begin(self, batch, logs=None):
        """Adjust learning rate at each batch."""
        cycle = np.floor(1 + self.iterations / (2 * self.step_size))
        x = np.abs(self.iterations / self.step_size - 2 * cycle + 1)
        new_lr = self.min_lr + (self.max_lr - self.min_lr) * np.maximum(0, (1 - x))

        # ✅ Fix: Ensure learning rate is properly assigned
        if hasattr(self.model.optimizer, "learning_rate"):
            self.model.optimizer.learning_rate.assign(new_lr)  # ✅ Corrected method
        else:
            logger.error("❌ Optimizer does not support dynamic learning rate changes.")

        self.history.setdefault("lr", []).append(new_lr)
        self.iterations += 1


class MonteCarloDropout(Layer):
    """Monte Carlo Dropout: Applies Dropout During Inference."""

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        return (
            tf.nn.dropout(inputs, rate=self.rate) if training else inputs
        )  # ✅ Apply dropout only during training

    def compute_output_shape(self, input_shape):
        return input_shape  # ✅ Ensure TensorFlow knows the output shape

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config


class LSTMStockPredictor:
    """
    Trains an improved LSTM model for stock price prediction.
    """

    def __init__(self, tickers=None, plot=False, lookback=30, random_state=42):
        self.tickers = tickers if tickers else self.fetch_tickers()
        self.plot = plot  # ✅ Store plot option
        self.lookback = lookback
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
        """Prepares time-series features and target for LSTM."""
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

        # ✅ New Advanced Features
        df["bollinger_upper"] = df["sma_50"] + (df["close"].rolling(20).std() * 2)
        df["bollinger_lower"] = df["sma_50"] - (df["close"].rolling(20).std() * 2)
        df["momentum_10"] = df["close"].diff(10)
        df["volatility_10"] = df["close"].pct_change().rolling(10).std()
        df["rsi_7"] = df["rsi_14"].rolling(7).mean()  # 🔥 Shorter-term RSI
        df["macd_hist"] = (
            df["macd"] - df["macd_signal"]
        )  # 🔥 MACD Histogram (Momentum Indicator)

        features += [
            "bollinger_upper",
            "bollinger_lower",
            "momentum_10",
            "volatility_10",
            "rsi_7",
            "macd_hist",
        ]

        # ✅ Ensure no NaN values
        df.fillna(method="ffill", inplace=True)  # ✅ Forward fill missing values
        df.fillna(method="bfill", inplace=True)  # ✅ Backward fill remaining NaNs
        df.dropna(inplace=True)  # ✅ Drop any remaining NaNs

        # ✅ Check for NaNs after cleaning
        if df[features].isna().sum().sum() > 0:
            logger.error("❌ NaN values still exist after preprocessing.")

        # ✅ Normalize Features
        scaled_data = self.scaler.fit_transform(df[features + [target]])

        X, y, dates = [], [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback : i])
            y.append(scaled_data[i, -1])
            dates.append(df.index[i])

        X, y = np.array(X), np.array(y)

        # ✅ Final NaN check before training
        if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
            logger.error("❌ NaN values detected in input data. Skipping training.")
            return None, None, None

        return X, y, dates

    @handle_exceptions
    def build_lstm_model(self, input_shape):
        """Builds an optimized LSTM model with stronger L2 regularization."""
        inputs = Input(shape=input_shape)

        lstm_out = LSTM(128, return_sequences=False, kernel_regularizer=l2(0.002))(
            inputs
        )  # 🔥 Stronger Regularization
        lstm_out = LayerNormalization()(lstm_out)  # 🔥 Normalize activations

        dense_out = Dense(64, activation="relu", kernel_regularizer=l2(0.002))(
            lstm_out
        )  # 🔥 More Regularization
        outputs = Dense(1, kernel_regularizer=l2(0.002))(dense_out)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)  # type: ignore
        return model

    @handle_exceptions
    def train_model(self, X, y):
        """Trains the LSTM model using a larger batch size for better stability."""
        model = self.build_lstm_model((X.shape[1], X.shape[2]))

        if model is None:
            logger.error("❌ Model failed to build. Skipping training.")
            return None

        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn(
                "[cyan]Training LSTM: [progress.percentage]{task.percentage:>3.1f}%"
            ),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Training LSTM...", total=250)

            class ProgressCallback(tf.keras.callbacks.Callback):  # type: ignore
                def on_epoch_end(self, epoch, logs=None):
                    progress.update(task, advance=1)

            optimizer = AdamW(learning_rate=0.0003, weight_decay=1e-3)

            model.compile(optimizer=optimizer, loss="mse")

            history = model.fit(
                X,
                y,
                epochs=250,
                batch_size=128,  # 🔥 Increased Batch Size
                validation_split=0.1,
                verbose=0,
                callbacks=[
                    ProgressCallback(),
                    EarlyStopping(
                        monitor="val_loss", patience=10, restore_best_weights=True
                    ),
                ],
            )

            progress.update(task, completed=250)

        # ✅ Extract final loss values
        final_train_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]

        # ✅ Display Final Training Summary in a Rich Table
        table = Table(
            title="[bold cyan]Training Summary for LSTM[/bold cyan]", show_lines=True
        )
        table.add_column("[bold yellow]Metric[/bold yellow]", justify="center")
        table.add_column("[bold green]Value[/bold green]", justify="center")

        table.add_row(
            "Final Training Loss", f"[bold green]{final_train_loss:.4f}[/bold green]"
        )
        table.add_row(
            "Final Validation Loss", f"[bold blue]{final_val_loss:.4f}[/bold blue]"
        )

        console.print(table)

        return model

    @handle_exceptions
    def evaluate_model(self, model, X, y, dates, ticker):
        """Evaluates model performance and logs results."""
        predictions = model.predict(X)

        # ✅ Check if model output contains NaN values
        if np.isnan(predictions).sum() > 0:
            logger.error(
                f"❌ Model predictions contain NaN values for {ticker}. Skipping evaluation."
            )
            return None, None

        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)

        table = Table(
            title=f"[bold cyan]LSTM Model Performance for {ticker}[/bold cyan]",
            show_lines=True,
        )
        table.add_column("[bold yellow]Metric[/bold yellow]", justify="center")
        table.add_column("[bold green]Value[/bold green]", justify="center")

        table.add_row("RMSE", f"[bold green]{rmse:.4f}[/bold green]")
        table.add_row("R² Score", f"[bold blue]{r2:.4f}[/bold blue]")

        console.print(table)

        logger.info(f"RMSE: {rmse:.4f}, R² Score: {r2:.4f}")

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
        """Saves trained LSTM model only if valid."""
        if model is None:  # ✅ Check for valid model before saving
            logger.error(
                f"❌ Skipping model save for {ticker} due to training failure."
            )
            return

        model_path = os.path.join(MODEL_DIR, f"lstm_{ticker}.keras")
        model.save(model_path)
        logger.info(f"✅ Model saved to {model_path}")

    @handle_exceptions
    def run_pipeline(self):
        """Executes the full training pipeline."""
        console.print(
            Panel(
                f"🚀 [bold cyan]Training LSTM Model for {len(self.tickers)} tickers...[/bold cyan]",
                style="cyan",
            )
        )
        logger.info(f"🚀 Training LSTM Model for {len(self.tickers)} tickers")

        for ticker in self.tickers:
            df = self.fetch_data(ticker)
            X, y, dates = self.prepare_features(df)
            model = self.train_model(X, y)
            self.evaluate_model(model, X, y, dates, ticker)
            self.save_model(model, ticker)

        console.print(
            Panel(
                "✅ [bold green]All LSTM models trained successfully![/bold green]",
                style="green",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LSTM model for stock price prediction."
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

    predictor = LSTMStockPredictor(
        tickers=args.ticker.split(",") if args.ticker else None, plot=args.plot
    )
    predictor.run_pipeline()
