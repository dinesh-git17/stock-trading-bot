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
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tqdm import tqdm

# Suppress TensorFlow logs & warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Load environment variables
load_dotenv()

# Setup logging
LOG_FILE = "data/logs/stock_gpt_training.log"
os.makedirs("data/logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
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

# Transformer Configuration
SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 50
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
DROPOUT_RATE = 0.2


def signal_handler(sig, frame):
    console.print("\n[bold red]❌ Process interrupted. Exiting safely...[/bold red]")
    logging.info("Process interrupted by user.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def count_saved_models():
    """Counts and logs the total number of trained models."""
    num_models = len([f for f in os.listdir(MODEL_PATH) if f.endswith(".keras")])
    console.print(
        f"[bold cyan]📊 Total StockGPT models saved: {num_models}[/bold cyan]"
    )
    logging.info(f"Total StockGPT models saved: {num_models}")
    return num_models


def load_data(ticker):
    """Loads preprocessed stock data for training/testing."""
    train_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_train.csv")
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        logging.warning(f"No data found for {ticker}. Skipping training.")
        return None, None

    return pd.read_csv(train_file), pd.read_csv(test_file)


def prepare_data(df_train, df_test, ticker):
    """Prepares stock data for Transformer training with matching feature sets."""

    df_train = df_train.select_dtypes(include=[np.number])
    df_test = df_test.select_dtypes(include=[np.number])

    scaler_file = os.path.join(SCALER_PATH, f"{ticker}_scaler.pkl")

    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        expected_features = (
            scaler.feature_names_in_
        )  # 🔥 Auto-detects required features
    else:
        expected_features = df_train.columns.tolist()
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

    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)

    def create_sequences(data):
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(data)):
            X.append(data[i - SEQUENCE_LENGTH : i, :])
            y.append(data[i, list(expected_features).index("close")])
        return np.array(X), np.array(y).reshape(-1, 1)

    X_train, y_train = create_sequences(df_train)
    X_test, y_test = create_sequences(df_test)

    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        logging.warning(f"{ticker} - Not enough data for training. Skipping...")
        console.print(f"[bold red]⚠ {ticker} - Not enough data. Skipping...[/bold red]")
        return None, None, None, None

    return X_train, y_train, X_test, y_test


def transformer_encoder(inputs):
    """Defines a single Transformer encoder block."""
    attn_output = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=D_MODEL)(
        inputs, inputs
    )
    attn_output = Dropout(DROPOUT_RATE)(attn_output)
    attn_output = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = Dense(FF_DIM, activation="relu")(attn_output)
    ffn_output = Dense(D_MODEL)(ffn_output)
    ffn_output = Dropout(DROPOUT_RATE)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(attn_output + ffn_output)


def build_transformer_model(input_shape):
    """Builds and compiles a Transformer model for stock price prediction."""
    inputs = Input(shape=input_shape)
    x = Dense(D_MODEL, activation="relu")(inputs)
    x = transformer_encoder(x)
    x = Conv1D(filters=32, kernel_size=3, activation="relu")(x)
    x = Flatten()(x)
    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(args):
    """Loads data, trains Transformer model, and saves best model."""
    ticker = args

    try:
        df_train, df_test = load_data(ticker)
        if df_train is None or df_test is None:
            return ticker, "❌ No Data"

        X_train, y_train, X_test, y_test = prepare_data(df_train, df_test, ticker)
        if X_train is None:
            return ticker, "❌ Insufficient Data"

        model = build_transformer_model((SEQUENCE_LENGTH, X_train.shape[2]))
        model_path = os.path.join(MODEL_PATH, f"stock_gpt_{ticker}.keras")

        model_checkpoint = ModelCheckpoint(
            model_path, save_best_only=True, monitor="val_loss", mode="min", verbose=0
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=0
        )

        model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[early_stopping, model_checkpoint],
        )

        model.save(model_path, save_format="keras")

        return ticker, "✅ Model Trained & Saved"

    except Exception as e:
        return ticker, f"❌ Training Failed: {str(e)}"


def process_all_models():
    """Trains all stock models in parallel and logs results."""
    tickers = [
        f.split("_")[0]
        for f in os.listdir(PROCESSED_DATA_PATH)
        if f.endswith("_train.csv")
    ]

    console.print(
        f"[bold cyan]🚀 Training StockGPT models for {len(tickers)} tickers...[/bold cyan]"
    )

    with Pool(min(8, cpu_count())) as pool:
        results = list(
            tqdm(
                pool.imap(train_model, tickers),
                total=len(tickers),
                desc="Training Progress",
            )
        )

    table = Table(title="StockGPT Training Results")
    table.add_column("Ticker", style="cyan", justify="center")
    table.add_column("Status", style="green", justify="center")

    for ticker, status in results:
        table.add_row(ticker, status)

    console.print(table)
    count_saved_models()


if __name__ == "__main__":
    process_all_models()
