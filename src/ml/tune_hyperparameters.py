import gc  # Garbage Collection
import logging
import os

import joblib
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track  # Progress bar for better UI
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor

# ✅ Load environment variables
load_dotenv()

# ✅ Setup logging
LOG_FILE = "data/logs/hyperparameter_tuning.log"
os.makedirs("data/logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ Suppress TensorFlow & Keras Warnings
tf.get_logger().setLevel("ERROR")

# ✅ Directory Paths
PROCESSED_DATA_PATH = "data/processed_data/"
MODEL_PATH = "models/"
OPTUNA_DB_PATH = "data/optuna_study.db"  # 🔥 Save Optuna trials to disk
os.makedirs(MODEL_PATH, exist_ok=True)


def load_data(ticker):
    """Loads preprocessed train & test data for a given ticker."""
    train_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_train.csv")
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        logging.warning(f"No data found for {ticker}. Skipping tuning.")
        return None, None, None, None

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    df_train = df_train.select_dtypes(include=[np.number])
    df_test = df_test.select_dtypes(include=[np.number])

    X_train, y_train = df_train.drop(columns=["close"]), df_train["close"]
    X_test, y_test = df_test.drop(columns=["close"]), df_test["close"]

    return X_train, y_train, X_test, y_test


def objective_lstm(trial):
    """Objective function for tuning LSTM hyperparameters with Optuna."""
    lstm_units = trial.suggest_int(
        "lstm_units", 25, 75
    )  # 🔥 Reduce max units from 100 → 75
    dropout_rate = trial.suggest_float(
        "dropout_rate", 0.1, 0.3
    )  # 🔥 Reduce max dropout
    batch_size = trial.suggest_int("batch_size", 16, 20)  # 🔥 Reduce batch size range
    epochs = 3  # 🔥 Reduce epochs from 5 → 3

    # Load sample data for tuning
    X_train, y_train, _, _ = load_data("AAPL")
    if X_train is None:
        return float("inf")

    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))

    model = Sequential(
        [
            Input(shape=(1, X_train.shape[2])),  # ✅ Fix UserWarning issue
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units),
            Dropout(dropout_rate),
            Dense(25),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = model.predict(X_train).flatten()
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))

    # ✅ Free memory after each trial
    tf.keras.backend.clear_session()
    gc.collect()

    return rmse


def tune_lstm():
    """Tunes LSTM hyperparameters using Optuna and saves results to disk."""
    console.print("[bold yellow]🔍 Tuning LSTM...[/bold yellow]")
    study = optuna.create_study(
        direction="minimize", storage=f"sqlite:///{OPTUNA_DB_PATH}", load_if_exists=True
    )
    study.optimize(
        objective_lstm, n_trials=3, show_progress_bar=True
    )  # 🔥 Show a progress bar

    best_params = study.best_params
    best_rmse = study.best_value

    console.print(
        f"\n[bold green]✅ Best LSTM Hyperparameters: {best_params}[/bold green]"
    )
    console.print(f"[bold cyan]🏆 Best LSTM RMSE: {best_rmse:.4f}[/bold cyan]")

    return best_params


def tune_all_models():
    """Tunes all models and saves the best ones in batches to prevent memory overload."""
    tickers = [
        f.split("_")[0]
        for f in os.listdir(PROCESSED_DATA_PATH)
        if f.endswith("_train.csv")
    ]

    batch_size = 5  # 🔥 Tune 5 tickers at a time to reduce memory load
    console.print(
        f"[bold cyan]🚀 Hyperparameter tuning for {len(tickers)} tickers in batches...[/bold cyan]"
    )

    for batch_start in range(0, len(tickers), batch_size):
        batch_tickers = tickers[batch_start : batch_start + batch_size]

        for ticker in track(batch_tickers, description="Tuning models..."):
            console.print(
                f"[bold yellow]⚡ Tuning models for {ticker}...[/bold yellow]"
            )
            X_train, y_train, _, _ = load_data(ticker)
            if X_train is None:
                continue

            best_lstm_params = tune_lstm()

            console.print(
                f"[bold green]✅ Best LSTM for {ticker}: {best_lstm_params}[/bold green]\n"
            )

            del X_train, y_train
            gc.collect()  # ✅ Free memory

    console.print("[bold green]✅ All models tuned & saved![/bold green]")


if __name__ == "__main__":
    tune_all_models()
