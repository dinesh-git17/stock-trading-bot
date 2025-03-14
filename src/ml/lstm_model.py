import logging
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import sqrt

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# ✅ Suppress TensorFlow Warnings & Logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# ✅ Setup Logging
LOG_FILE = "data/logs/lstm_training.log"
os.makedirs("data/logs", exist_ok=True)

# ✅ Insert blank lines before logging new logs
with open(LOG_FILE, "a") as log_file:
    log_file.write("\n" * 3)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ Directories
DATA_DIR = "data/transformed"
SCALER_DIR = "data/transformed/scalers"
MODEL_DIR = "models"
LSTM_PRED_DIR = "data/lstm_predictions"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LSTM_PRED_DIR, exist_ok=True)

# ✅ LSTM Configuration
LOOKBACK_DAYS = 100
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_WORKERS = min(4, os.cpu_count())


def load_data(ticker):
    """Loads transformed stock data and MinMaxScaler for a given ticker."""
    data_path = f"{DATA_DIR}/transformed_{ticker}.pkl"
    scaler_path = f"{SCALER_DIR}/scaler_{ticker}.pkl"

    if not os.path.exists(data_path) or not os.path.exists(scaler_path):
        return None, None, None

    with open(data_path, "rb") as f:
        X, Y = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return X, Y, scaler


def build_lstm_model(input_shape):
    """Builds an improved LSTM model for stock price prediction."""
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(150, return_sequences=True),
            Dropout(0.4),
            LSTM(150, return_sequences=False),
            Dropout(0.4),
            Dense(50, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model


def train_and_save_lstm_predictions(ticker):
    """Trains the LSTM model and saves predictions for XGBoost."""

    X, Y, scaler = load_data(ticker)
    if X is None:
        return ticker, None

    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)

    X_train, Y_train = X[:train_size], Y[:train_size]
    X_val, Y_val = (
        X[train_size : train_size + val_size],
        Y[train_size : train_size + val_size],
    )
    X_test, Y_test = X[train_size + val_size :], Y[train_size + val_size :]

    X_train = np.reshape(
        X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2])
    )
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    tf.keras.backend.clear_session()

    model = build_lstm_model((LOOKBACK_DAYS, X.shape[2]))
    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    # ✅ Save LSTM Model
    model_path = f"{MODEL_DIR}/lstm_{ticker}.h5"
    model.save(model_path)

    # ✅ Generate LSTM Predictions for XGBoost
    Y_pred = model.predict(X, verbose=0)

    # ✅ Save LSTM Predictions
    lstm_pred_path = f"{LSTM_PRED_DIR}/lstm_{ticker}.pkl"
    with open(lstm_pred_path, "wb") as f:
        pickle.dump(Y_pred, f)

    logging.info(f"✅ Trained LSTM model saved for {ticker} -> {model_path}")
    logging.info(f"📊 LSTM Predictions saved for {ticker} -> {lstm_pred_path}")

    return ticker


if __name__ == "__main__":
    console.print(
        "[bold cyan]🚀 Training LSTM models and saving predictions for XGBoost...[/bold cyan]"
    )

    tickers = [
        f.split("_")[-1].split(".")[0]
        for f in os.listdir(DATA_DIR)
        if f.startswith("transformed_")
    ]

    # ✅ Allow user to specify number of tickers to evaluate via CLI
    if len(sys.argv) > 1:
        try:
            num_tickers = int(sys.argv[1])
            tickers = tickers[:num_tickers]
            console.print(
                f"[bold yellow]🔹 Evaluating only {num_tickers} tickers as specified.[/bold yellow]"
            )
        except ValueError:
            console.print(
                "[bold red]❌ ERROR: Invalid number of tickers. Using all available.[/bold red]"
            )

    # ✅ Rich Progress Bar for Training
    with Progress(
        TextColumn("[bold blue]⏳ Training LSTM Model:[/bold blue] {task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing Tickers...", total=len(tickers))

        results = []
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_ticker = {
                executor.submit(train_and_save_lstm_predictions, ticker): ticker
                for ticker in tickers
            }

            for future in as_completed(future_to_ticker):
                ticker = future.result()
                if ticker:
                    results.append(ticker)
                    progress.update(task, advance=1)

    console.print(
        "\n[bold green]✅ LSTM training complete! Predictions saved for XGBoost.[/bold green]\n"
    )
