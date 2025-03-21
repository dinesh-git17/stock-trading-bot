import argparse
import logging
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model  # type: ignore

from src.tools.utils import get_database_engine, handle_exceptions, setup_logging

# ✅ Setup logging
LOG_FILE = "data/logs/benchmark_models.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)


# ✅ Suppress TensorFlow GPU logs and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress INFO, WARNING, ERROR logs
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
tf.get_logger().setLevel("ERROR")  # Suppress internal TensorFlow logging

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ✅ Rich console setup
console = Console()
logger.info("🚀 Benchmark script started.")
console.print(
    Panel("🚀 [bold green]Benchmark Script Started[/bold green]", style="bold green")
)

# ✅ Constants
MODEL_DIR = "models/"
CSV_FALLBACK_DIR = "data/pre_processed/"
LOOKBACK = 30

# ✅ Feature sets
XGB_FEATURES = [
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

LSTM_FEATURES = XGB_FEATURES + [
    "macd_hist",
    "bollinger_upper",
    "bollinger_lower",
    "momentum_10",
    "volatility_10",
    "rsi_7",
    "adjusted_close",
]


@handle_exceptions
def load_data(ticker):
    """Load processed stock data from DB or fallback CSV."""
    engine = get_database_engine()
    query = f"SELECT * FROM processed_stock_data WHERE ticker = '{ticker}'"

    console.print(
        Panel(f"📥 Loading data for [cyan]{ticker}[/cyan]...", style="bold blue")
    )

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            logger.info(f"✅ Loaded {len(df)} rows for {ticker} from database.")
            console.print(f"✅ [green]Loaded {len(df)} rows from database.[/green]")
    except Exception as e:
        logger.warning(f"⚠ Database error: {e}. Trying fallback CSV.")
        fallback_path = os.path.join(CSV_FALLBACK_DIR, f"{ticker}.csv")
        console.print(
            f"⚠ [yellow]Database failed. Trying fallback at {fallback_path}[/yellow]"
        )
        if os.path.exists(fallback_path):
            df = pd.read_csv(fallback_path)
            logger.info(f"✅ Loaded {len(df)} rows from fallback CSV for {ticker}.")
            console.print(f"✅ [green]Loaded {len(df)} rows from fallback CSV.[/green]")
        else:
            logger.error(f"❌ No fallback CSV found at {fallback_path}.")
            console.print(f"❌ [red]Failed: No fallback CSV found for {ticker}[/red]")
            return None

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by="date", inplace=True)
    df.set_index("date", inplace=True)
    return df


@handle_exceptions
def preprocess_data(df):
    """Preprocess data for LSTM and XGBoost models, including computed features."""
    console.print(Panel("🧼 Preprocessing data...", style="bold blue"))

    df = df.copy()
    df.drop(columns=["ticker"], errors="ignore", inplace=True)

    # ✅ Compute engineered features used by LSTM
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["bollinger_upper"] = df["sma_50"] + df["close"].rolling(20).std() * 2
    df["bollinger_lower"] = df["sma_50"] - df["close"].rolling(20).std() * 2
    df["momentum_10"] = df["close"].diff(10)
    df["volatility_10"] = df["close"].pct_change().rolling(10).std()
    df["rsi_7"] = df["rsi_14"].rolling(7).mean()

    # ✅ Clean NaNs
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.dropna(inplace=True)

    if df.empty:
        logger.error("❌ Dataframe is empty after preprocessing.")
        console.print("❌ [red]DataFrame is empty after preprocessing.[/red]")
        return None, None, None, None, None

    # ✅ Scale separately for each model
    xgb_scaler = StandardScaler()
    lstm_scaler = StandardScaler()

    X_xgb_scaled = xgb_scaler.fit_transform(df[XGB_FEATURES])
    y_xgb = lstm_scaler.fit_transform(df[["close"]]).flatten()

    lstm_data = lstm_scaler.fit_transform(df[LSTM_FEATURES + ["close"]])
    dates = df.index

    # ✅ Reshape LSTM inputs
    X_lstm, y_lstm, lstm_dates = [], [], []
    for i in range(LOOKBACK, len(lstm_data)):
        X_lstm.append(lstm_data[i - LOOKBACK : i, :-1])
        y_lstm.append(lstm_data[i, -1])
        lstm_dates.append(dates[i])

    console.print(
        f"✅ [green]Preprocessing complete. LSTM samples: {len(X_lstm)}[/green]"
    )
    return (
        np.array(X_xgb_scaled[LOOKBACK:]),
        np.array(y_xgb[LOOKBACK:]),
        np.array(X_lstm),
        np.array(y_lstm),
        lstm_dates,
    )


@handle_exceptions
def load_models(ticker):
    """Load LSTM and XGBoost models from disk."""
    console.print(Panel("📦 Loading models from disk...", style="bold blue"))

    lstm_path = os.path.join(MODEL_DIR, f"lstm_{ticker}.keras")
    xgb_path = os.path.join(MODEL_DIR, f"xgboost_{ticker}.pkl")

    if not os.path.exists(lstm_path):
        logger.error(f"❌ LSTM model not found at {lstm_path}")
        console.print(f"❌ [red]LSTM model not found for {ticker}[/red]")
        return None, None
    if not os.path.exists(xgb_path):
        logger.error(f"❌ XGBoost model not found at {xgb_path}")
        console.print(f"❌ [red]XGBoost model not found for {ticker}[/red]")
        return None, None

    lstm_model = load_model(lstm_path)
    xgb_model = joblib.load(xgb_path)

    logger.info("✅ Both models loaded successfully.")
    console.print("✅ [green]Both models loaded successfully.[/green]")
    return lstm_model, xgb_model


@handle_exceptions
def evaluate_models(
    ticker, lstm_model, xgb_model, X_lstm, y_lstm, X_xgb, y_xgb, dates, plot
):
    """Run predictions and evaluate both models."""
    console.print(Panel("🧠 Evaluating models...", style="bold blue"))

    lstm_pred = lstm_model.predict(X_lstm).flatten()
    xgb_pred = xgb_model.predict(X_xgb).flatten()

    lstm_rmse = np.sqrt(mean_squared_error(y_lstm, lstm_pred))
    lstm_r2 = r2_score(y_lstm, lstm_pred)

    xgb_rmse = np.sqrt(mean_squared_error(y_xgb, xgb_pred))
    xgb_r2 = r2_score(y_xgb, xgb_pred)

    table = Table(title=f"📊 Model Benchmark for {ticker}", show_lines=True)
    table.add_column("Metric", style="cyan", justify="center")
    table.add_column("LSTM", style="green", justify="center")
    table.add_column("XGBoost", style="yellow", justify="center")
    table.add_row("RMSE", f"{lstm_rmse:.4f}", f"{xgb_rmse:.4f}")
    table.add_row("R² Score", f"{lstm_r2:.4f}", f"{xgb_r2:.4f}")
    console.print(table)

    winner = "LSTM" if lstm_rmse < xgb_rmse and lstm_r2 > xgb_r2 else "XGBoost"
    console.print(
        Panel(
            f"✅ [bold green]{winner}[/bold green] outperforms the other model based on RMSE and R² Score.",
            style="bold green",
        )
    )

    if plot:
        console.print("📈 [cyan]Plotting predictions...[/cyan]")
        plt.figure(figsize=(12, 5))
        plt.plot(dates, y_lstm, label="Actual", color="blue")
        plt.plot(dates, lstm_pred, label="LSTM Pred", color="red", linestyle="--")
        plt.plot(dates, xgb_pred, label="XGBoost Pred", color="orange", linestyle="--")
        plt.title(f"{ticker} - Actual vs Predicted (LSTM & XGBoost)")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


@handle_exceptions
def run_benchmark(ticker, plot=False):
    """Main function to run benchmarking pipeline."""
    console.print(
        Panel(
            f"🚀 Benchmarking models for [bold cyan]{ticker}[/bold cyan]...",
            style="bold cyan",
        )
    )

    df = load_data(ticker)
    if df is None:
        return

    X_xgb, y_xgb, X_lstm, y_lstm, dates = preprocess_data(df)
    if X_lstm is None or X_xgb is None:
        return

    lstm_model, xgb_model = load_models(ticker)
    if lstm_model is None or xgb_model is None:
        return

    evaluate_models(
        ticker, lstm_model, xgb_model, X_lstm, y_lstm, X_xgb, y_xgb, dates, plot
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark LSTM vs XGBoost for a ticker."
    )
    parser.add_argument(
        "--ticker", required=True, type=str, help="Ticker symbol to benchmark."
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show plots for prediction comparison."
    )
    args = parser.parse_args()

    run_benchmark(ticker=args.ticker, plot=args.plot)
