import logging
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

# ✅ Load environment variables
load_dotenv()

# ✅ Setup logging
LOG_FILE = "data/logs/compare_models.log"
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
RESULTS_PATH = "data/comparisons/"
os.makedirs(RESULTS_PATH, exist_ok=True)


def load_test_data(ticker):
    """Loads the test data for a given ticker."""
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")

    if not os.path.exists(test_file):
        console.print(
            f"[bold red]⚠ No test data found for {ticker}. Skipping...[/bold red]"
        )
        return None

    return pd.read_csv(test_file).select_dtypes(include=[np.number])


def compute_sharpe_ratio(returns, risk_free_rate=0.02):
    """Computes the Sharpe Ratio for a given model's returns."""
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0  # Avoid division by zero
    return np.mean(excess_returns) / np.std(excess_returns)


def make_predictions(model, X_test):
    """Predicts stock prices using a given model."""
    return model.predict(X_test)


def evaluate_predictions(y_true, y_pred):
    """Computes RMSE, MAE, and Sharpe Ratio."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    returns = np.diff(y_pred) / y_pred[:-1]  # Simulated returns
    sharpe_ratio = compute_sharpe_ratio(returns)

    return rmse, mae, sharpe_ratio


def compare_models(ticker):
    """Compares LSTM, XGBoost, Random Forest, and Hybrid models for a given ticker."""
    console.print(f"[bold yellow]🔍 Comparing models for {ticker}...[/bold yellow]")
    df_test = load_test_data(ticker)
    if df_test is None:
        return None

    X_test = df_test.drop(columns=["close"])
    y_test = df_test["close"].values

    models = {}
    results = {}

    # ✅ Load Models
    try:
        models["LSTM"] = load_model(
            os.path.join(MODEL_PATH, f"hybrid_lstm_{ticker}.h5")
        )
        models["XGBoost"] = joblib.load(
            os.path.join(MODEL_PATH, f"tuned_xgboost_{ticker}.joblib")
        )
        models["RandomForest"] = joblib.load(
            os.path.join(MODEL_PATH, f"tuned_random_forest_{ticker}.joblib")
        )
        models["Hybrid"] = joblib.load(
            os.path.join(MODEL_PATH, f"hybrid_xgboost_{ticker}.joblib")
        )
    except Exception as e:
        console.print(f"[bold red]⚠ Error loading models for {ticker}: {e}[/bold red]")
        return None

    # ✅ Evaluate Each Model
    for model_name, model in models.items():
        try:
            y_pred = make_predictions(model, X_test)
            rmse, mae, sharpe = evaluate_predictions(y_test, y_pred)
            results[model_name] = {"RMSE": rmse, "MAE": mae, "Sharpe Ratio": sharpe}
        except Exception as e:
            console.print(
                f"[bold red]⚠ Error in {model_name} for {ticker}: {e}[/bold red]"
            )

    # ✅ Select Best Model (Lowest RMSE)
    best_model = min(results, key=lambda k: results[k]["RMSE"])

    return {
        "Ticker": ticker,
        "Best Model": best_model,
        **results[best_model],  # Add best model's RMSE, MAE, Sharpe Ratio
    }


def compare_all_models():
    """Runs comparison for all tickers and saves results."""
    tickers = [
        f.split("_")[0]
        for f in os.listdir(PROCESSED_DATA_PATH)
        if f.endswith("_test.csv")
    ]

    console.print(
        f"[bold cyan]🚀 Comparing models for {len(tickers)} tickers...[/bold cyan]"
    )
    results = []

    for ticker in track(tickers, description="Comparing models..."):
        result = compare_models(ticker)
        if result:
            results.append(result)

    results_df = pd.DataFrame(results)
    results_file = os.path.join(RESULTS_PATH, "comparison_results.csv")
    results_df.to_csv(results_file, index=False)

    console.print(
        f"[bold green]✅ Model comparison complete! Results saved to {results_file}[/bold green]"
    )


if __name__ == "__main__":
    compare_all_models()
