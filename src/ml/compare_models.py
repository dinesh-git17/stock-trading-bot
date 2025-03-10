import multiprocessing
import os

# ✅ Fix multiprocessing crash on macOS
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn", force=True)

import logging
import signal
import sys
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from tqdm import tqdm

# ✅ Suppress TensorFlow logs & Metal plugin logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = (
    "0"  # Disable unnecessary TensorFlow optimizations
)
tf.get_logger().setLevel("ERROR")

# ✅ Load environment variables
load_dotenv()

# ✅ Setup logging
LOG_FILE = "data/logs/compare_models.log"
os.makedirs("data/logs", exist_ok=True)
os.makedirs("data/comparisons", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ Directory Paths
PROCESSED_DATA_PATH = "data/processed_data/"
MODEL_PATH = "models/"
SCALER_PATH = "models/scalers/"
SEQUENCE_LENGTH = 60


def signal_handler(sig, frame):
    console.print("\n[bold red]❌ Process interrupted. Exiting safely...[/bold red]")
    logging.info("Process interrupted by user.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def load_test_data(ticker):
    """Loads test data and scaler, ensuring feature consistency."""
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")
    scaler_file = os.path.join(SCALER_PATH, f"{ticker}_scaler.pkl")

    if not os.path.exists(test_file) or not os.path.exists(scaler_file):
        logging.warning(f"{ticker} - No test data or scaler found. Skipping...")
        return None, None

    df_test = pd.read_csv(test_file).select_dtypes(include=[np.number])
    scaler = joblib.load(scaler_file)

    # 🔥 Ensure test data has the same features as training
    expected_features = scaler.feature_names_in_
    for col in expected_features:
        if col not in df_test.columns:
            df_test[col] = 0  # Fill missing columns with 0s
    df_test = df_test[expected_features]  # Ensure correct column order

    return df_test, scaler


def make_predictions(model, X_test, scaler, feature_columns):
    """Makes predictions and scales back to original values."""
    predictions = model.predict(
        X_test.astype(np.float32), verbose=0
    )  # ✅ Suppressed output

    predictions_padded = np.hstack(
        [predictions] + [np.zeros_like(predictions)] * (len(feature_columns) - 1)
    )
    predictions_rescaled = scaler.inverse_transform(predictions_padded)[:, 0]

    return predictions_rescaled


def evaluate_predictions(y_true, y_pred):
    """Computes RMSE."""
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def compare_models(ticker):
    """Compares LSTM vs. StockGPT for a given ticker."""
    logging.info(f"{ticker} - Comparing models...")

    df_test, scaler = load_test_data(ticker)
    if df_test is None:
        return None

    feature_columns = df_test.columns.tolist()

    # ✅ Ensure model file paths exist
    lstm_model_path = os.path.join(MODEL_PATH, f"lstm_{ticker}.h5")
    stock_gpt_model_path = os.path.join(MODEL_PATH, f"stock_gpt_{ticker}.keras")

    if not os.path.exists(lstm_model_path) or not os.path.exists(stock_gpt_model_path):
        logging.warning(f"{ticker} - Models not found. Skipping...")
        return None

    # ✅ Load LSTM Model
    lstm_model = load_model(lstm_model_path, compile=False)

    # ✅ Load StockGPT Model (with error handling)
    try:
        stock_gpt_model = tf.keras.models.load_model(
            stock_gpt_model_path, compile=False
        )
    except Exception as e:
        logging.error(f"{ticker} - Model Loading Failed: {e}")
        return None

    # ✅ Prepare Test Data
    X_test = np.array(
        [
            df_test.iloc[i - SEQUENCE_LENGTH : i].values
            for i in range(SEQUENCE_LENGTH, len(df_test))
        ]
    )
    y_test_actual = df_test.iloc[SEQUENCE_LENGTH:]["close"].values

    # ✅ Make Predictions
    lstm_rmse = evaluate_predictions(
        y_test_actual, make_predictions(lstm_model, X_test, scaler, feature_columns)
    )
    stock_gpt_rmse = evaluate_predictions(
        y_test_actual,
        make_predictions(stock_gpt_model, X_test, scaler, feature_columns),
    )

    # ✅ Determine the Better Model
    better_model = (
        "LSTM"
        if lstm_rmse < stock_gpt_rmse
        else "StockGPT" if stock_gpt_rmse < lstm_rmse else "Tie"
    )

    return {
        "ticker": ticker,
        "lstm_rmse": lstm_rmse,
        "stock_gpt_rmse": stock_gpt_rmse,
        "better_model": better_model,
    }


def compare_all_models():
    """Runs comparison for all tickers and saves results."""
    tickers = [
        f.split("_")[0]
        for f in os.listdir(PROCESSED_DATA_PATH)
        if f.endswith("_test.csv")
    ]

    console.print(
        f"[bold cyan]🚀 Comparing StockGPT vs. LSTM for {len(tickers)} tickers...[/bold cyan]"
    )

    # ✅ Cleaner progress bar (prevents multiple print lines)
    results = []
    with Pool(min(8, cpu_count())) as pool:
        for result in tqdm(
            pool.imap(compare_models, tickers),
            total=len(tickers),
            desc="Comparison Progress",
            dynamic_ncols=True,
        ):
            if result:
                results.append(result)

    results_df = pd.DataFrame(results)

    # ✅ Compute Overall Best Model
    best_model_counts = results_df["better_model"].value_counts()
    overall_best = (
        best_model_counts.idxmax() if not best_model_counts.empty else "No Data"
    )

    # ✅ Append Overall Result Row
    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                [
                    {
                        "ticker": "Overall",
                        "lstm_rmse": "-",
                        "stock_gpt_rmse": "-",
                        "better_model": overall_best,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    # ✅ Save Results to CSV
    results_filename = f"data/comparisons/comparison_results_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
    results_df.to_csv(results_filename, index=False)

    console.print(
        f"[bold green]✅ Model comparison complete! Results saved to {results_filename}[/bold green]"
    )


if __name__ == "__main__":
    compare_all_models()
