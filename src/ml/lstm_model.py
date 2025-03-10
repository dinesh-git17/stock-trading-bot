import multiprocessing
import os

# ✅ Fix multiprocessing crash on macOS
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn", force=True)

import logging
import signal
import sys
import warnings
from multiprocessing import Pool, cpu_count

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tqdm import tqdm

# Suppress TensorFlow Metal plugin logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/lstm_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

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
    console.print("\n[bold red]❌ Process interrupted. Exiting safely...[/bold red]")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def count_saved_models():
    """Counts and prints the number of saved models."""
    num_models = len([f for f in os.listdir(MODEL_PATH) if f.endswith(".h5")])
    console.print(f"[bold cyan]📊 Total models saved: {num_models}[/bold cyan]")
    return num_models


def load_data(ticker):
    """Loads preprocessed training and test data for a ticker."""
    train_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_train.csv")
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        return None, None

    return pd.read_csv(train_file), pd.read_csv(test_file)


def prepare_data(df_train, df_test, ticker):
    """Prepares train and test data by scaling and creating sequences."""

    df_train = df_train.drop(columns=["date"], errors="ignore")
    df_test = df_test.drop(columns=["date"], errors="ignore")

    scaler_file = os.path.join(SCALER_PATH, f"{ticker}_scaler.pkl")

    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        expected_features = scaler.feature_names_in_  # 🔥 Auto-detects feature names
    else:
        # If scaler doesn't exist, use train columns as expected features
        expected_features = df_train.select_dtypes(include=[np.number]).columns.tolist()
        scaler = MinMaxScaler()
        scaler.fit(df_train[expected_features])
        joblib.dump(scaler, scaler_file)

    # 🔥 Ensure train & test have the same feature names
    for col in expected_features:
        if col not in df_train.columns:
            df_train[col] = 0  # Fill missing columns with 0s
        if col not in df_test.columns:
            df_test[col] = 0  # Fill missing columns with 0s

    # Ensure column order matches scaler expectation
    df_train = df_train[expected_features]
    df_test = df_test[expected_features]

    df_train[expected_features] = scaler.transform(df_train[expected_features])
    df_test[expected_features] = scaler.transform(df_test[expected_features])

    def create_sequences(data):
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(data)):
            X.append(data[i - SEQUENCE_LENGTH : i, :])
            y.append(data[i, list(expected_features).index("close")])
        return np.array(X), np.array(y).reshape(-1, 1)

    X_train, y_train = create_sequences(df_train.values)
    X_test, y_test = create_sequences(df_test.values)

    return X_train, y_train, X_test, y_test


def build_lstm_model(input_shape):
    """Builds a bidirectional LSTM model with explicit Input() layer."""
    inputs = Input(shape=input_shape)

    x = Bidirectional(LSTM(units=128, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)

    x = Bidirectional(LSTM(units=128, return_sequences=False))(x)
    x = Dropout(0.2)(x)

    x = Dense(units=64, activation="relu")(x)
    outputs = Dense(units=1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def train_model(args):
    """Trains an LSTM model for a given ticker."""
    ticker, _ = args

    try:
        df_train, df_test = load_data(ticker)
        if df_train is None or df_test is None:
            return ticker, "❌ Skipped (No Data)"

        X_train, y_train, X_test, y_test = prepare_data(df_train, df_test, ticker)
        if len(X_train) == 0 or len(X_test) == 0:
            return ticker, "❌ Skipped (Insufficient Data)"

        model = build_lstm_model((SEQUENCE_LENGTH, X_train.shape[2]))
        model_checkpoint_path = os.path.join(MODEL_PATH, f"lstm_{ticker}.h5")

        callbacks = [
            ModelCheckpoint(
                model_checkpoint_path,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                verbose=0,
            ),
            EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
            ),
        ]

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
            callbacks=callbacks,
        )

        return ticker, "✅ Model Trained & Saved"

    except Exception as e:
        return ticker, f"❌ Training Failed: {str(e)}"


def process_all_models():
    """Trains LSTM models for all available tickers in parallel."""
    tickers = [
        f.split("_")[0]
        for f in os.listdir(PROCESSED_DATA_PATH)
        if f.endswith("_train.csv")
    ]

    console.print(
        f"[bold cyan]🚀 Training LSTM models for {len(tickers)} tickers...[/bold cyan]"
    )

    with Pool(min(8, cpu_count())) as pool:
        results = list(
            tqdm(
                pool.imap(train_model, [(ticker, None) for ticker in tickers]),
                total=len(tickers),
                desc="Training Progress",
            )
        )

    # Display summary table
    table = Table(title="LSTM Training Results")
    table.add_column("Ticker", style="cyan", justify="center")
    table.add_column("Status", style="green", justify="center")

    for ticker, status in results:
        table.add_row(ticker, status)

    console.print(table)
    console.print("[bold green]✅ All LSTM model training complete![/bold green]")
    count_saved_models()


if __name__ == "__main__":
    process_all_models()
