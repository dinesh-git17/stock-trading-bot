import multiprocessing
import os

# ✅ Fix multiprocessing crash on macOS
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn", force=True)

import logging
from multiprocessing import Pool, cpu_count

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ✅ Load environment variables
load_dotenv()

# ✅ Setup logging
LOG_FILE = "data/logs/random_forest_training.log"
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


def train_random_forest(X_train, y_train, use_parallel=True):
    """Trains a Random Forest model on stock data."""
    model = RandomForestRegressor(
        n_estimators=300,  # 🔥 Optimized hyperparameters
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1 if use_parallel else 1,  # ✅ Fix multiprocessing conflict
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and logs performance metrics."""
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    logging.info(f"Random Forest RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return rmse, mae


def train_single_ticker(ticker):
    """Trains a Random Forest model for a single ticker (to be used in multiprocessing)."""
    X_train, y_train, X_test, y_test = load_data(ticker)
    if X_train is None:
        return ticker, "❌ Skipped (No Data)"

    model = train_random_forest(
        X_train, y_train, use_parallel=False
    )  # ✅ Avoid parallel conflict
    rmse, mae = evaluate_model(model, X_test, y_test)

    model_file = os.path.join(MODEL_PATH, f"random_forest_{ticker}.joblib")
    joblib.dump(model, model_file)

    logging.info(f"Random Forest model saved for {ticker}")
    return ticker, f"✅ Trained & Saved (RMSE: {rmse:.4f}, MAE: {mae:.4f})"


def process_all_tickers():
    """Trains Random Forest models for all available tickers using parallel processing."""
    tickers = [
        f.split("_")[0]
        for f in os.listdir(PROCESSED_DATA_PATH)
        if f.endswith("_train.csv")
    ]

    console.print(
        f"[bold cyan]🚀 Training Random Forest for {len(tickers)} tickers...[/bold cyan]"
    )

    results = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing tickers...", total=len(tickers))

        # ✅ Use multiprocessing but disable parallelism inside RandomForest
        with Pool(min(8, cpu_count())) as pool:
            for result in pool.imap(train_single_ticker, tickers):
                if result:
                    results.append(result)
                progress.update(task, advance=1)

    # ✅ Display Training Results
    table = Table(title="Random Forest Training Results")
    table.add_column("Ticker", style="cyan", justify="center")
    table.add_column("Status", style="green", justify="center")

    for ticker, status in results:
        table.add_row(ticker, status)

    console.print(table)
    console.print(
        "[bold green]✅ All Random Forest models trained and saved![/bold green]"
    )


if __name__ == "__main__":
    process_all_tickers()
