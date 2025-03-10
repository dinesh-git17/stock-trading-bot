import multiprocessing
import os

# ✅ Fix multiprocessing crash on macOS
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn", force=True)

import logging
import warnings
from multiprocessing import Pool, cpu_count

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ✅ Hide TensorFlow verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# ✅ Load environment variables
load_dotenv()

# ✅ Setup logging
LOG_FILE = "data/logs/hybrid_model_training.log"
os.makedirs("data/logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ Directory Paths
PROCESSED_DATA_PATH = "data/processed_data/"
MODEL_PATH = "models/"
os.makedirs(MODEL_PATH, exist_ok=True)


def load_data(ticker):
    """Loads preprocessed train & test data for a given ticker."""
    train_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_train.csv")
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        logging.warning(f"No data found for {ticker}. Skipping training.")
        return None, None, None, None

    df_train = pd.read_csv(train_file).select_dtypes(include=[np.number])
    df_test = pd.read_csv(test_file).select_dtypes(include=[np.number])

    # ✅ Ensure test data has the same features as training
    expected_features = df_train.columns.tolist()
    for col in expected_features:
        if col not in df_test.columns:
            df_test[col] = 0  # Fill missing columns with 0s

    # ✅ Split into Features (X) and Target (y)
    X_train, y_train = df_train.drop(columns=["close"]), df_train["close"]
    X_test, y_test = df_test.drop(columns=["close"]), df_test["close"]

    return X_train, y_train, X_test, y_test


def build_lstm_model(input_shape):
    """Builds and compiles an optimized LSTM model for stock prediction without warnings."""
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm(X_train, y_train):
    """Trains an LSTM model and returns predictions for XGBoost."""
    # ✅ Reshape data for LSTM (samples, time steps, features)
    X_train_reshaped = np.reshape(
        X_train.values, (X_train.shape[0], 1, X_train.shape[1])
    )

    model = build_lstm_model((1, X_train.shape[1]))
    model.fit(
        X_train_reshaped, y_train, epochs=30, batch_size=32, verbose=0
    )  # ✅ Silent training

    # ✅ Generate LSTM predictions (to be used as XGBoost feature)
    lstm_predictions = model.predict(X_train_reshaped, verbose=0)
    return lstm_predictions.flatten(), model


def train_xgboost(X_train, y_train):
    """Trains an optimized XGBoost model using both original features and LSTM predictions."""
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0,  # ✅ Disable XGBoost logs
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and logs performance metrics."""
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    logging.info(f"Hybrid Model RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return rmse, mae


def train_single_ticker(ticker):
    """Trains the Hybrid Model (LSTM + XGBoost) for a single ticker."""
    X_train, y_train, X_test, y_test = load_data(ticker)
    if X_train is None:
        return ticker, "❌ Skipped (No Data)"

    # ✅ Train LSTM and get predictions
    lstm_predictions, lstm_model = train_lstm(X_train, y_train)

    # ✅ Add LSTM predictions as a new feature
    X_train["lstm_pred"] = lstm_predictions

    # ✅ Train XGBoost with enhanced features
    xgb_model = train_xgboost(X_train, y_train)

    # ✅ Process test data for evaluation
    lstm_test_predictions = lstm_model.predict(
        np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1])),
        verbose=0,
    )
    X_test["lstm_pred"] = lstm_test_predictions.flatten()

    rmse, mae = evaluate_model(xgb_model, X_test, y_test)

    # ✅ Save both models
    lstm_model.save(os.path.join(MODEL_PATH, f"hybrid_lstm_{ticker}.h5"))
    joblib.dump(xgb_model, os.path.join(MODEL_PATH, f"hybrid_xgboost_{ticker}.joblib"))

    logging.info(f"Hybrid Model saved for {ticker}")
    return ticker, f"✅ Trained & Saved (RMSE: {rmse:.4f}, MAE: {mae:.4f})"


def process_all_tickers():
    """Trains Hybrid Models for all available tickers using parallel processing."""
    tickers = [
        f.split("_")[0]
        for f in os.listdir(PROCESSED_DATA_PATH)
        if f.endswith("_train.csv")
    ]

    console.print(
        f"[bold cyan]🚀 Training Hybrid Model for {len(tickers)} tickers...[/bold cyan]"
    )

    results = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing tickers...", total=len(tickers))

        # ✅ Use multiprocessing for faster training
        with Pool(min(8, cpu_count())) as pool:
            for result in pool.imap(train_single_ticker, tickers):
                if result:
                    results.append(result)
                progress.update(task, advance=1)

    # ✅ Display Training Results
    table = Table(title="Hybrid Model Training Results")
    table.add_column("Ticker", style="cyan", justify="center")
    table.add_column("Status", style="green", justify="center")

    for ticker, status in results:
        table.add_row(ticker, status)

    console.print(table)
    console.print("[bold green]✅ All Hybrid Models trained and saved![/bold green]")


if __name__ == "__main__":
    process_all_tickers()
