import gc  # Memory Cleanup
import logging
import multiprocessing
import os
import signal
import sys
import time
from multiprocessing import Manager, Pool, cpu_count

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
)
from tensorflow.keras.models import Sequential

# ✅ Set multiprocessing start method for macOS compatibility
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn", force=True)

# ✅ Suppress TensorFlow Metal plugin logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# ✅ Load environment variables
load_dotenv()

# ✅ Setup logging
LOG_FILE = "data/logs/lstm_training.log"
os.makedirs("data/logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)

console = Console()

# ✅ Directory Paths
PROCESSED_DATA_PATH = "data/processed_data/"
MODEL_PATH = "models/"
SCALER_PATH = "models/scalers/"
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(SCALER_PATH, exist_ok=True)

SEQUENCE_LENGTH = 60  # Adjustable sequence length


def signal_handler(sig, frame):
    """Handles user interruption (Ctrl+C)."""
    console.print("\n[bold red]❌ Process interrupted. Cleaning up...[/bold red]")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def get_valid_tickers():
    """Finds all valid tickers that have both train & test data."""
    all_files = os.listdir(PROCESSED_DATA_PATH)
    tickers = set()

    for filename in all_files:
        if filename.endswith("_train.csv"):
            ticker = filename.replace("_train.csv", "")
            if f"{ticker}_test.csv" in all_files:
                tickers.add(ticker)  # ✅ Ensure test data exists before adding

    return sorted(list(tickers))


def load_data(ticker):
    """Loads preprocessed training and test data for a ticker."""
    train_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_train.csv")
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        return None, None

    df_train, df_test = pd.read_csv(train_file), pd.read_csv(test_file)
    return df_train, df_test


def prepare_data(df_train, df_test, ticker):
    """Prepares train and test data by scaling and creating sequences."""
    df_train = df_train.select_dtypes(include=[np.number])
    df_test = df_test.select_dtypes(include=[np.number])

    # ✅ Drop columns that are entirely NaN
    df_train.dropna(axis=1, how="all", inplace=True)
    df_test.dropna(axis=1, how="all", inplace=True)

    # ✅ Fill remaining NaNs with median values to avoid errors
    df_train.fillna(df_train.median(), inplace=True)
    df_test.fillna(df_test.median(), inplace=True)

    if df_train.isnull().sum().sum() > 0 or df_test.isnull().sum().sum() > 0:
        logging.warning(f"⚠ {ticker} contains NaN even after imputation! Skipping.")
        return None, None, None, None

    scaler_file = os.path.join(SCALER_PATH, f"{ticker}_scaler.pkl")

    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        expected_features = scaler.feature_names_in_
    else:
        expected_features = df_train.columns.tolist()
        scaler = MinMaxScaler()
        scaler.fit(df_train)
        joblib.dump(scaler, scaler_file)

    # ✅ Ensure test data has the same features as train
    for col in expected_features:
        if col not in df_test.columns:
            df_test[col] = 0  # Fill missing features with zeros

    df_test = df_test[expected_features]

    df_train = pd.DataFrame(scaler.transform(df_train), columns=expected_features)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=expected_features)

    def create_sequences(data):
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(data)):
            X.append(data[i - SEQUENCE_LENGTH : i, :])
            y.append(data[i, list(expected_features).index("close")])
        return np.array(X), np.array(y).reshape(-1, 1)

    return create_sequences(df_train.values) + create_sequences(df_test.values)


def build_lstm_model(input_shape):
    """Builds an optimized LSTM model."""
    model = Sequential(
        [
            Input(shape=input_shape),
            LayerNormalization(),
            Bidirectional(LSTM(units=128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(units=128, return_sequences=False)),
            Dropout(0.3),
            Dense(units=64, activation="relu"),
            Dense(units=1),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(ticker, log_queue):
    """Trains an LSTM model for a given ticker with clear console logging."""
    start_time = time.time()

    console.print(f"[bold cyan]🚀 Processing {ticker}...[/bold cyan]")

    df_train, df_test = load_data(ticker)
    if df_train is None or df_test is None:
        log_queue.put(f"{ticker}|❌ No Data|0.00|N/A")
        console.print(f"[red]⚠ No data available for {ticker}. Skipping.[/red]")
        return

    console.print(f"[bold yellow]🔄 Preprocessing & Scaling {ticker}...[/bold yellow]")
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test, ticker)
    if X_train is None or X_test is None:
        log_queue.put(f"{ticker}|❌ Insufficient Data|0.00|N/A")
        console.print(f"[red]⚠ Not enough valid data for {ticker}. Skipping.[/red]")
        return

    console.print(f"[bold blue]🛠 Building LSTM Model for {ticker}...[/bold blue]")
    model = build_lstm_model(input_shape=(SEQUENCE_LENGTH, X_train.shape[2]))

    console.print(f"[bold green]🎯 Training LSTM Model for {ticker}...[/bold green]")
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    )

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0,
        callbacks=[early_stopping, lr_scheduler],
    )

    model_path = os.path.join(MODEL_PATH, f"lstm_{ticker}.h5")
    model.save(model_path)

    console.print(f"[bold green]✅ Training Completed for {ticker}![/bold green]")


def main():
    """Runs the LSTM training pipeline using multiprocessing."""
    tickers = get_valid_tickers()

    if len(sys.argv) > 1:
        try:
            num_tickers = int(sys.argv[1])
            tickers = tickers[:num_tickers]
        except ValueError:
            console.print(
                "[red]❌ Invalid argument. Please provide a valid number.[/red]"
            )
            sys.exit(1)

    console.print(
        f"\n[bold cyan]🚀 Training LSTM models for {len(tickers)} tickers...[/bold cyan]\n"
    )

    with Pool(min(8, cpu_count())) as pool:
        pool.starmap(train_model, [(ticker, Manager().Queue()) for ticker in tickers])

    console.print("\n[bold green]✅ All models trained successfully![/bold green]\n")


if __name__ == "__main__":
    main()
