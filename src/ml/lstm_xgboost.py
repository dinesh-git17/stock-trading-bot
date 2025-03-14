import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Setup Logging
LOG_FILE = "data/logs/lstm_xgboost.log"
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
LSTM_PRED_DIR = "data/lstm_predictions"
MODEL_DIR = "models"
XGB_MODEL_DIR = "models/xgboost"
EVALUATION_DIR = "data/evaluation"
os.makedirs(XGB_MODEL_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)

# ✅ Configuration
TEST_SPLIT = 0.1  # Use 10% of the data for testing


def load_xgb_data(ticker):
    """Loads dataset and LSTM predictions for XGBoost training."""
    data_path = f"{DATA_DIR}/transformed_{ticker}.pkl"
    lstm_pred_path = f"{LSTM_PRED_DIR}/lstm_{ticker}.pkl"

    if not os.path.exists(data_path) or not os.path.exists(lstm_pred_path):
        logging.warning(f"⚠️ Data or LSTM predictions missing for {ticker}. Skipping...")
        return None, None, None

    # ✅ Load Features (OHLCV + Technical Indicators)
    with open(data_path, "rb") as f:
        X, Y = pickle.load(f)

    # ✅ Load LSTM Predictions
    with open(lstm_pred_path, "rb") as f:
        lstm_preds = pickle.load(f)

    # ✅ Fix Shape Mismatch (Flatten Time Series Data)
    X = X.reshape(
        X.shape[0], -1
    )  # Convert (samples, time_steps, features) -> (samples, features)

    # ✅ Fix Shape Mismatch for LSTM Predictions
    lstm_preds = lstm_preds.reshape(
        lstm_preds.shape[0], 1
    )  # Ensure it has shape (samples, 1)

    # ✅ Concatenate LSTM Predictions as a Feature
    X = np.hstack((X, lstm_preds))

    return X, Y


def train_xgboost(ticker):
    """Trains XGBoost model using LSTM predictions as a feature."""

    X, Y = load_xgb_data(ticker)
    if X is None:
        return ticker, None

    # ✅ Split Data
    train_size = int(len(X) * (1 - TEST_SPLIT))
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]

    # ✅ Train XGBoost Model
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(X_train, Y_train)

    # ✅ Make Predictions
    Y_pred = model.predict(X_test)

    # ✅ Compute Metrics
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    # ✅ Save XGBoost Model
    model_path = f"{XGB_MODEL_DIR}/xgboost_{ticker}.json"
    model.save_model(model_path)

    logging.info(f"✅ Trained XGBoost model saved for {ticker} -> {model_path}")
    logging.info(f"📊 {ticker} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # ✅ Save Plot
    save_plot(ticker, Y_test, Y_pred)

    return ticker, rmse, mae, r2


def save_plot(ticker, Y_actual, Y_pred):
    """Plots and saves Actual vs. Predicted stock prices for XGBoost."""
    plt.figure(figsize=(10, 5))
    plt.plot(Y_actual, label="Actual Price", color="blue")
    plt.plot(Y_pred, label="Predicted Price", linestyle="dashed", color="red")
    plt.title(f"{ticker} - XGBoost: Actual vs. Predicted Stock Prices")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid()

    plot_path = f"{EVALUATION_DIR}/{ticker}_xgboost_evaluation.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"📊 XGBoost Evaluation plot saved: {plot_path}")


if __name__ == "__main__":
    console.print(
        "[bold cyan]🚀 Training XGBoost models using LSTM predictions...[/bold cyan]"
    )

    tickers = [
        f.split("_")[-1].split(".")[0]
        for f in os.listdir(LSTM_PRED_DIR)
        if f.startswith("lstm_")
    ]

    # ✅ CLI Argument for Specifying Number of Tickers to Train
    if len(sys.argv) > 1:
        try:
            num_tickers = int(sys.argv[1])
            tickers = tickers[:num_tickers]
            console.print(
                f"[bold yellow]🔹 Training only {num_tickers} tickers as specified.[/bold yellow]"
            )
        except ValueError:
            console.print(
                "[bold red]❌ ERROR: Invalid number of tickers. Using all available.[/bold red]"
            )

    # ✅ Rich Progress Bar for Training
    with Progress(
        TextColumn(
            "[bold blue]⏳ Training XGBoost Model:[/bold blue] {task.description}"
        ),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing Tickers...", total=len(tickers))

        results = []
        for ticker in tickers:
            ticker, rmse, mae, r2 = train_xgboost(ticker)
            if rmse is not None:
                results.append((ticker, rmse, mae, r2))
            progress.update(task, advance=1)

    # ✅ Print Evaluation Metrics in a Rich Table
    table = Table(title="📊 XGBoost Model Evaluation Results")
    table.add_column("Ticker", justify="center", style="cyan")
    table.add_column("RMSE", justify="center", style="green")
    table.add_column("MAE", justify="center", style="magenta")
    table.add_column("R² Score", justify="center", style="yellow")

    for ticker, rmse, mae, r2 in results:
        table.add_row(ticker, f"{rmse:.4f}", f"{mae:.4f}", f"{r2:.4f}")

    console.print(
        "\n[bold green]✅ XGBoost training complete! All results displayed below.[/bold green]\n"
    )
    console.print(table)
