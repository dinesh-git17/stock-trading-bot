import numpy as np
import pandas as pd
import os
import logging
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from multiprocessing import Pool, cpu_count, Manager, set_start_method
import signal
import sys

# ✅ Fix multiprocessing issue on macOS
if sys.platform == "darwin":
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

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# TensorFlow Suppression
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TensorFlow logs

# Directory Paths
PROCESSED_DATA_PATH = "data/processed_data/"
MODEL_PATH = "models/"
os.makedirs(MODEL_PATH, exist_ok=True)  # ✅ Ensure model directory exists

# LSTM Configuration
SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 50

# ✅ Use Manager() to share results & skipped tickers across processes
manager = Manager()
results = manager.list()
skipped_tickers = manager.list()
processed_tickers = manager.list()  # ✅ Track processed tickers to prevent duplicates


# Handle Ctrl+C to exit safely
def signal_handler(sig, frame):
    console.print(
        "\n[bold red]❌ Process interrupted. Exiting safely...[/bold red]",
        justify="center",
    )
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def count_saved_models():
    """Counts and logs the total number of models saved."""
    num_models = len([f for f in os.listdir(MODEL_PATH) if f.endswith(".h5")])
    console.print(f"[bold cyan]📊 Total models saved: {num_models}[/bold cyan]")
    return num_models


def load_data(ticker):
    """Loads preprocessed training and testing data for a given ticker."""
    train_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_train.csv")
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        logging.warning(f"No data found for {ticker}. Skipping training.")
        skipped_tickers.append((ticker, "No Data"))
        return None, None

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    return df_train, df_test


def prepare_data(df_train, df_test, ticker):
    """Prepares data for LSTM model (reshapes into 3D format)."""
    df_train = df_train.drop(columns=["date"], errors="ignore")
    df_test = df_test.drop(columns=["date"], errors="ignore")

    # Ensure only numeric columns remain
    feature_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()

    # Scale only numeric columns
    scaler = MinMaxScaler()
    df_train[feature_columns] = scaler.fit_transform(df_train[feature_columns])
    df_test[feature_columns] = scaler.transform(df_test[feature_columns])

    def create_sequences(data):
        """Creates LSTM-compatible sequences."""
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(data)):
            X.append(data[i - SEQUENCE_LENGTH : i, :])  # Past 60 days as input
            y.append(
                data[i, feature_columns.index("close")]
            )  # Get the index of 'close' column

        return np.array(X), np.array(y).reshape(
            -1, 1
        )  # Reshape y for LSTM compatibility

    X_train, y_train = create_sequences(df_train[feature_columns].values)
    X_test, y_test = create_sequences(df_test[feature_columns].values)

    return X_train, y_train, X_test, y_test, scaler, feature_columns


def build_lstm_model(input_shape):
    """Builds and compiles an optimized LSTM model."""
    model = Sequential(
        [
            LSTM(units=64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=64, return_sequences=False),
            Dropout(0.2),
            Dense(units=1),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(ticker):
    """Loads data, trains LSTM model, and saves best model."""
    if ticker in processed_tickers:
        console.print(
            f"[bold blue]{ticker:<6} 🔄 Skipping duplicate training.[/bold blue]"
        )
        return

    console.print(f"[bold yellow]{ticker:<6} ⏳ Training model...[/bold yellow]")

    try:
        df_train, df_test = load_data(ticker)
        if df_train is None or df_test is None:
            return

        X_train, y_train, X_test, y_test, scaler, feature_columns = prepare_data(
            df_train, df_test, ticker
        )

        if len(X_train) == 0 or len(X_test) == 0:
            skipped_tickers.append((ticker, "Insufficient Data"))
            return

        model = build_lstm_model((SEQUENCE_LENGTH, X_train.shape[2]))

        # ✅ Ensure model file path
        model_checkpoint_path = os.path.join(MODEL_PATH, f"lstm_{ticker}.h5")

        # Callbacks for early stopping & saving best model
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

        model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[early_stopping],
        )

        # ✅ Save model at the end
        model.save(model_checkpoint_path)
        processed_tickers.append(ticker)  # ✅ Mark as processed

        console.print(f"[bold green]{ticker:<6} ✅ Model trained & saved.[/bold green]")

    except Exception as e:
        console.print(f"[bold red]{ticker:<6} ❌ Training failed: {e}[/bold red]")
        skipped_tickers.append((ticker, "Training Failed"))


def process_all_models():
    """Processes all tickers using parallel processing."""
    tickers = [
        f.split("_")[0]
        for f in os.listdir(PROCESSED_DATA_PATH)
        if f.endswith("_train.csv")
    ]

    console.print(
        f"[bold cyan]🚀 Training LSTM models for {len(tickers)} tickers in parallel...[/bold cyan]",
        justify="left",
    )

    with Pool(cpu_count()) as pool:
        pool.map(train_model, tickers)

    console.print(
        "[bold green]✅ All LSTM model training complete![/bold green]", justify="left"
    )

    # ✅ Count and log total saved models
    count_saved_models()


if __name__ == "__main__":
    process_all_models()
