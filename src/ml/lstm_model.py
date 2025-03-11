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
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
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

SEQUENCE_LENGTH = 60


def signal_handler(sig, frame):
    """Handles user interruption (Ctrl+C)."""
    console.print("\n[bold red]❌ Process interrupted. Cleaning up...[/bold red]")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def log_message(msg, log_queue):
    """Thread-safe logging with a multiprocessing queue."""
    log_queue.put(msg)


def process_logs(log_queue):
    """Handles structured logging from multiple processes cleanly."""
    console.print("\n[bold cyan]📊 Training Progress:[/bold cyan]\n")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Ticker", justify="center", style="cyan")
    table.add_column("Status", justify="center", style="green")
    table.add_column("Training Time", justify="center", style="yellow")
    table.add_column("Model Path", justify="left", style="white")

    logs = []
    while True:
        msg = log_queue.get()
        if msg == "STOP":
            break
        logs.append(msg)

    for log in logs:
        ticker, status, elapsed_time, model_path = log.split("|")
        table.add_row(ticker, status, f"{elapsed_time}s", model_path)

    console.print(table)


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

    scaler_file = os.path.join(SCALER_PATH, f"{ticker}_scaler.pkl")

    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        expected_features = scaler.feature_names_in_
    else:
        expected_features = df_train.columns.tolist()
        scaler = MinMaxScaler()
        scaler.fit(df_train)
        joblib.dump(scaler, scaler_file)

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


def train_model(ticker, log_queue):
    """Trains an LSTM model for a given ticker."""
    start_time = time.time()

    df_train, df_test = load_data(ticker)
    if df_train is None or df_test is None:
        log_queue.put(f"{ticker}|❌ No Data|0.00|N/A")
        return

    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test, ticker)
    if len(X_train) == 0 or len(X_test) == 0:
        log_queue.put(f"{ticker}|❌ Insufficient Data|0.00|N/A")
        return

    model = Sequential(
        [
            Input(shape=(SEQUENCE_LENGTH, X_train.shape[2])),
            Bidirectional(LSTM(units=100, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(units=100, return_sequences=False)),
            Dropout(0.2),
            Dense(units=64, activation="relu"),
            Dense(units=1),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    os.makedirs(MODEL_PATH, exist_ok=True)
    model_path = os.path.join(MODEL_PATH, f"lstm_{ticker}.h5")
    model.save(model_path)

    elapsed_time = round(time.time() - start_time, 2)
    log_queue.put(f"{ticker}|✅ Trained & Saved|{elapsed_time}|{model_path}")


if __name__ == "__main__":
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

    manager = Manager()
    log_queue = manager.Queue()

    log_process = multiprocessing.Process(target=process_logs, args=(log_queue,))
    log_process.start()

    with Pool(min(8, cpu_count())) as pool:
        pool.starmap(train_model, [(ticker, log_queue) for ticker in tickers])

    log_queue.put("STOP")
    log_process.join()

    console.print("\n[bold green]✅ All models trained successfully![/bold green]\n")
