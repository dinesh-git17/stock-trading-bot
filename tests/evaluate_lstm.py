import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# ✅ Suppress TensorFlow Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# ✅ Setup Logging
LOG_FILE = "data/logs/evaluate_lstm.log"
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
EVALUATION_DIR = "data/evaluation"
os.makedirs(EVALUATION_DIR, exist_ok=True)

# ✅ Configuration
LOOKBACK_DAYS = 30
LEARNING_RATE = 0.001


def load_test_data(ticker):
    """Loads test dataset and MinMaxScaler for a given ticker."""
    data_path = f"{DATA_DIR}/transformed_{ticker}.pkl"
    scaler_path = f"{SCALER_DIR}/scaler_{ticker}.pkl"

    if not os.path.exists(data_path) or not os.path.exists(scaler_path):
        logging.warning(f"⚠️ Data or scaler missing for {ticker}. Skipping...")
        return None, None, None

    with open(data_path, "rb") as f:
        X, Y = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return X, Y, scaler


def evaluate_lstm(ticker):
    """Evaluates the trained LSTM model for a given ticker."""

    # ✅ Load Data & Model
    X, Y, scaler = load_test_data(ticker)
    model_path = f"{MODEL_DIR}/lstm_{ticker}.h5"

    if X is None or not os.path.exists(model_path):
        return ticker, None, None, None  # Skip missing tickers

    # ✅ Load Trained Model (Fixed `mse` Error)
    model = load_model(model_path, compile=False)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")

    # ✅ Use only test set (last 10% of data)
    test_size = int(len(X) * 0.1)
    X_test, Y_test = X[-test_size:], Y[-test_size:]

    # ✅ Reshape Data for LSTM
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # ✅ Make Predictions
    Y_pred = model.predict(X_test, verbose=0)

    # ✅ Fix Inverse Transformation for Closing Price Only
    close_price_index = -1  # Assuming closing price is last column in scaled data
    Y_pred_rescaled = scaler.inverse_transform(
        np.concatenate(
            [np.zeros((Y_pred.shape[0], scaler.n_features_in_ - 1)), Y_pred], axis=1
        )
    )[:, close_price_index]

    Y_test_rescaled = scaler.inverse_transform(
        np.concatenate(
            [
                np.zeros((Y_test.shape[0], scaler.n_features_in_ - 1)),
                Y_test.reshape(-1, 1),
            ],
            axis=1,
        )
    )[:, close_price_index]

    # ✅ Compute Metrics
    rmse = np.sqrt(mean_squared_error(Y_test_rescaled, Y_pred_rescaled))
    mae = mean_absolute_error(Y_test_rescaled, Y_pred_rescaled)
    r2 = r2_score(Y_test_rescaled, Y_pred_rescaled)

    logging.info(
        f"✅ {ticker} Evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}"
    )

    return ticker, rmse, mae, r2


def save_plot(ticker, Y_actual, Y_pred):
    """Plots and saves Actual vs. Predicted stock prices."""
    plt.figure(figsize=(10, 5))
    plt.plot(Y_actual, label="Actual Price", color="blue")
    plt.plot(Y_pred, label="Predicted Price", linestyle="dashed", color="red")
    plt.title(f"{ticker} - Actual vs. Predicted Stock Prices")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid()

    plot_path = f"{EVALUATION_DIR}/{ticker}_evaluation.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"📊 Evaluation plot saved: {plot_path}")


if __name__ == "__main__":
    console.print("[bold cyan]📊 Evaluating LSTM model performance...[/bold cyan]")

    # ✅ Get all tickers & Parse CLI Argument
    tickers = [
        f.split("_")[-1].split(".")[0]
        for f in os.listdir(MODEL_DIR)
        if f.startswith("lstm_")
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

    # ✅ Rich Progress Bar for Evaluation
    with Progress(
        TextColumn("[bold blue]📊 Evaluating Model:[/bold blue] {task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing Tickers...", total=len(tickers))

        results = []
        for ticker in tickers:
            ticker, rmse, mae, r2 = evaluate_lstm(ticker)
            if rmse is not None:
                results.append((ticker, rmse, mae, r2))
            progress.update(task, advance=1)

    # ✅ Print Evaluation Metrics in a Rich Table
    table = Table(title="📊 LSTM Model Evaluation Results")
    table.add_column("Ticker", justify="center", style="cyan")
    table.add_column("RMSE", justify="center", style="green")
    table.add_column("MAE", justify="center", style="magenta")
    table.add_column("R² Score", justify="center", style="yellow")

    for ticker, rmse, mae, r2 in results:
        table.add_row(ticker, f"{rmse:.4f}", f"{mae:.4f}", f"{r2:.4f}")

    console.print(
        "\n[bold green]✅ Model Evaluation Complete! All results displayed below.[/bold green]\n"
    )
    console.print(table)
