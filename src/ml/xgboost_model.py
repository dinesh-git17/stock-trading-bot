import os
import joblib
import logging
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dotenv import load_dotenv
from rich.console import Console

# ✅ Load environment variables
load_dotenv()

# ✅ Setup logging
LOG_FILE = "data/logs/xgboost_training.log"
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

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # ✅ Use only numerical columns
    df_train = df_train.select_dtypes(include=[np.number])
    df_test = df_test.select_dtypes(include=[np.number])

    # ✅ Split into Features (X) and Target (y)
    X_train, y_train = df_train.drop(columns=["close"]), df_train["close"]
    X_test, y_test = df_test.drop(columns=["close"]), df_test["close"]

    return X_train, y_train, X_test, y_test


def train_xgboost(X_train, y_train):
    """Trains an XGBoost model on stock data."""
    model = XGBRegressor(
        n_estimators=200,  # 🔥 Later, we will tune this with GridSearchCV
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and logs performance metrics."""
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    console.print(f"[bold cyan]📊 XGBoost RMSE: {rmse:.4f}, MAE: {mae:.4f}[/bold cyan]")
    logging.info(f"XGBoost RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return rmse, mae


def train_and_save_xgboost(ticker):
    """Loads data, trains XGBoost, evaluates, and saves the model."""
    console.print(f"[bold yellow]⏳ Training XGBoost for {ticker}...[/bold yellow]")

    X_train, y_train, X_test, y_test = load_data(ticker)
    if X_train is None:
        console.print(f"[bold red]⚠ No data for {ticker}. Skipping.[/bold red]")
        return

    model = train_xgboost(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    model_file = os.path.join(MODEL_PATH, f"xgboost_{ticker}.joblib")
    joblib.dump(model, model_file)

    console.print(f"[bold green]✅ XGBoost model saved: {model_file}[/bold green]")
    logging.info(f"XGBoost model saved for {ticker}")


def process_all_tickers():
    """Trains XGBoost models for all available tickers."""
    tickers = [
        f.split("_")[0]
        for f in os.listdir(PROCESSED_DATA_PATH)
        if f.endswith("_train.csv")
    ]

    console.print(
        f"[bold cyan]🚀 Training XGBoost for {len(tickers)} tickers...[/bold cyan]"
    )

    for ticker in tickers:
        train_and_save_xgboost(ticker)

    console.print("[bold green]✅ All XGBoost models trained and saved![/bold green]")


if __name__ == "__main__":
    process_all_tickers()
