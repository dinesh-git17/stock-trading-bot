import numpy as np
import pandas as pd
import os
import logging
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from multiprocessing import Pool, cpu_count, Manager, Process
import joblib
import signal
import sys
from time import sleep

if sys.platform == "darwin":
    from multiprocessing import set_start_method

    set_start_method("fork")

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/lstm_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# Suppress warnings & TensorFlow logs
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel("ERROR")

# Directory Paths
PROCESSED_DATA_PATH = "data/processed_data/"
MODEL_PATH = "models/"
SCALER_PATH = "models/scalers/"
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(SCALER_PATH, exist_ok=True)

SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 50


def signal_handler(sig, frame):
    console.print(
        "\n[bold red]❌ Process interrupted. Exiting safely...[/bold red]",
        justify="left",
    )
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def count_saved_models():
    num_models = len([f for f in os.listdir(MODEL_PATH) if f.endswith(".h5")])
    console.print(f"[bold cyan]📊 Total models saved: {num_models}[/bold cyan]")
    return num_models


def load_data(ticker):
    train_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_train.csv")
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        return None, None

    return pd.read_csv(train_file), pd.read_csv(test_file)


def prepare_data(df_train, df_test, ticker):
    df_train = df_train.drop(columns=["date"], errors="ignore")
    df_test = df_test.drop(columns=["date"], errors="ignore")

    feature_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler_file = os.path.join(SCALER_PATH, f"{ticker}_scaler.pkl")

    scaler = MinMaxScaler()
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
    else:
        scaler.fit(df_train[feature_columns])
        joblib.dump(scaler, scaler_file)

    df_train[feature_columns] = scaler.transform(df_train[feature_columns])
    df_test[feature_columns] = scaler.transform(df_test[feature_columns])

    def create_sequences(data):
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(data)):
            X.append(data[i - SEQUENCE_LENGTH : i, :])
            y.append(data[i, feature_columns.index("close")])
        return np.array(X), np.array(y).reshape(-1, 1)

    X_train, y_train = create_sequences(df_train[feature_columns].values)
    X_test, y_test = create_sequences(df_test[feature_columns].values)

    return X_train, y_train, X_test, y_test, scaler, feature_columns


def build_lstm_model(input_shape):
    model = Sequential(
        [
            Bidirectional(
                LSTM(units=128, return_sequences=True), input_shape=input_shape
            ),
            Dropout(0.2),
            Bidirectional(LSTM(units=128, return_sequences=False)),
            Dropout(0.2),
            Dense(units=64, activation="relu"),
            Dense(units=1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(args):
    ticker, completed_jobs, log_queue = args

    try:
        log_queue.put(f"[bold yellow]{ticker:<6} ⏳ Training model...[/bold yellow]")

        df_train, df_test = load_data(ticker)
        if df_train is None or df_test is None:
            completed_jobs.append(ticker)
            return ticker, "❌ Skipped (No Data)"

        X_train, y_train, X_test, y_test, scaler, feature_columns = prepare_data(
            df_train, df_test, ticker
        )
        if len(X_train) == 0 or len(X_test) == 0:
            completed_jobs.append(ticker)
            return ticker, "❌ Skipped (Insufficient Data)"

        model = build_lstm_model((SEQUENCE_LENGTH, X_train.shape[2]))
        model_checkpoint_path = os.path.join(MODEL_PATH, f"lstm_{ticker}.h5")

        model_checkpoint = ModelCheckpoint(
            model_checkpoint_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=0,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(
            BATCH_SIZE
        )
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
            BATCH_SIZE
        )

        model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=test_dataset,
            verbose=0,
            callbacks=[early_stopping, model_checkpoint],
        )

        completed_jobs.append(ticker)
        log_queue.put(f"[bold green]{ticker:<6} ✅ Model Trained & Saved[/bold green]")
        return ticker, "✅ Model Trained & Saved"

    except Exception as e:
        completed_jobs.append(ticker)
        log_queue.put(f"[bold red]{ticker:<6} ❌ Training Failed: {str(e)}[/bold red]")
        return ticker, f"❌ Training Failed: {str(e)}"


def logger_process(log_queue):
    while True:
        message = log_queue.get()
        if message is None:
            break
        console.print(message)


def process_all_models():
    tickers = [
        f.split("_")[0]
        for f in os.listdir(PROCESSED_DATA_PATH)
        if f.endswith("_train.csv")
    ]

    console.print(
        f"[bold cyan]🚀 Training LSTM models for {len(tickers)} tickers...[/bold cyan]",
        justify="left",
    )

    manager = Manager()
    completed_jobs = manager.list()
    log_queue = manager.Queue()

    log_proc = Process(target=logger_process, args=(log_queue,))
    log_proc.start()

    with Pool(cpu_count()) as pool:
        results = pool.map(
            train_model, [(ticker, completed_jobs, log_queue) for ticker in tickers]
        )

    log_queue.put(None)  # Stop logger
    log_proc.join()

    table = Table(title="LSTM Training Results")
    table.add_column("Ticker", style="cyan", justify="center")
    table.add_column("Status", style="green", justify="center")

    for ticker, status in results:
        table.add_row(ticker, status)

    console.print(table)
    console.print(
        "[bold green]✅ All LSTM model training complete![/bold green]", justify="left"
    )
    count_saved_models()


if __name__ == "__main__":
    process_all_models()
