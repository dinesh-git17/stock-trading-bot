import os
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import warnings
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    MultiHeadAttention,
    Conv1D,
    LayerNormalization,
    Dropout,
    Dense,
    Flatten,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error  # ✅ FIXED IMPORT
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from multiprocessing import Pool, cpu_count
import signal
import sys

# ✅ Fix multiprocessing issue on macOS
if sys.platform == "darwin":
    from multiprocessing import set_start_method

    set_start_method("fork")

# ✅ Load environment variables
load_dotenv()

# ✅ Setup logging (logs both to file and console)
LOG_FILE = "data/logs/compare_models.log"
os.makedirs("data/logs", exist_ok=True)
os.makedirs("data/comparisons", exist_ok=True)  # ✅ Create comparison folder

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ Suppress warnings & TensorFlow logs
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel("ERROR")

# ✅ Directory Paths
PROCESSED_DATA_PATH = "data/processed_data/"
MODEL_PATH = "models/"
SCALER_PATH = "models/scalers/"


# ✅ Handle Ctrl+C safely
def signal_handler(sig, frame):
    console.print("\n[bold red]❌ Process interrupted. Exiting safely...[/bold red]")
    logging.info("Process interrupted by user.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def load_test_data(ticker):
    """Loads the test data and corresponding scaler for a given ticker."""
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")
    scaler_file = os.path.join(SCALER_PATH, f"{ticker}_scaler.pkl")

    if not os.path.exists(test_file) or not os.path.exists(scaler_file):
        console.print(
            f"[bold red]⚠ No test data or scaler found for {ticker}. Skipping...[/bold red]"
        )
        logging.warning(f"{ticker} - No test data or scaler found. Skipping...")
        return None, None

    df_test = pd.read_csv(test_file).select_dtypes(include=[np.number])
    scaler = joblib.load(scaler_file)

    return df_test, scaler


def make_predictions(model, X_test, scaler, feature_columns):
    """Makes predictions and scales back to original values."""
    predictions = model.predict(
        X_test.astype(np.float32), verbose=0
    )  # 🔥 Suppress TensorFlow logs

    # ✅ Ensure correct shape before inverse transformation
    predictions_padded = np.hstack(
        [predictions] + [np.zeros_like(predictions)] * (len(feature_columns) - 1)
    )
    predictions_rescaled = scaler.inverse_transform(predictions_padded)[:, 0]

    return predictions_rescaled


def evaluate_predictions(y_true, y_pred):
    """Computes RMSE."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse


def compare_models(ticker):
    """Compares LSTM vs. StockGPT for a given ticker."""
    console.print(f"[bold yellow]{ticker:<6} 🔍 Comparing models...[/bold yellow]")
    logging.info(f"{ticker} - Comparing models...")

    df_test, scaler = load_test_data(ticker)
    if df_test is None:
        return None

    feature_columns = df_test.columns.tolist()

    # ✅ Ensure model file paths exist
    lstm_model_path = os.path.join(MODEL_PATH, f"lstm_{ticker}.h5")
    stock_gpt_model_path = os.path.join(MODEL_PATH, f"stock_gpt_{ticker}.keras")

    if not os.path.exists(lstm_model_path) or not os.path.exists(stock_gpt_model_path):
        console.print(
            f"[bold red]⚠ Models not found for {ticker}. Skipping...[/bold red]"
        )
        logging.warning(f"{ticker} - Models not found. Skipping...")
        return None

    # ✅ Load LSTM Model
    lstm_model = load_model(lstm_model_path)

    # ✅ Load StockGPT Model
    try:
        stock_gpt_model = tf.keras.models.load_model(stock_gpt_model_path)
    except Exception as e:
        console.print(f"[bold red]❌ {ticker} - Model Loading Failed: {e}[/bold red]")
        logging.error(f"{ticker} - Model Loading Failed: {e}")
        return None

    # ✅ Prepare Test Data
    X_test = np.array(
        [df_test.iloc[i - 60 : i].values for i in range(60, len(df_test))]
    )
    y_test_actual = scaler.inverse_transform(
        np.hstack(
            [df_test.iloc[60:]["close"].values.reshape(-1, 1)]
            + [np.zeros((len(df_test) - 60, 1))] * (len(feature_columns) - 1)
        )
    )[:, 0]

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

    with Pool(cpu_count()) as pool:
        results = pool.map(compare_models, tickers)

    results = [r for r in results if r is not None]
    results_df = pd.DataFrame(results)

    # ✅ Compute Overall Best Model
    best_model_counts = results_df["better_model"].value_counts()
    overall_best = best_model_counts.idxmax()

    # ✅ Append Overall Result Row (🔥 Fixed `append` issue)
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
