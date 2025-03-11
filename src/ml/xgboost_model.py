import json
import logging
import os
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb  # ✅ Needed for callbacks
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track
from rich.table import Table

# ✅ Required for HalvingGridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import HalvingGridSearchCV
from xgboost import XGBRegressor

# ✅ Load environment variables
load_dotenv()

# ✅ Setup logging
LOG_DIR = "data/logs"
LOG_FILE = os.path.join(LOG_DIR, "xgboost_training.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()
errors = []  # ✅ Collect errors here for summary

# ✅ Directory Paths
PROCESSED_DATA_PATH = "data/processed_data/"
MODEL_PATH = "models/"
os.makedirs(MODEL_PATH, exist_ok=True)


def load_data(ticker):
    """Loads preprocessed train & test data for a given ticker."""
    try:
        train_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_train.csv")
        test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Data not found for {ticker}")

        df_train = pd.read_csv(train_file).select_dtypes(include=[np.number])
        df_test = pd.read_csv(test_file).select_dtypes(include=[np.number])

        # ✅ Ensure that test data has the same feature columns as train data
        common_features = list(set(df_train.columns) & set(df_test.columns))
        df_train = df_train[common_features]
        df_test = df_test[common_features]

        X_train, y_train = df_train.drop(columns=["close"]), df_train["close"]
        X_test, y_test = df_test.drop(columns=["close"]), df_test["close"]

        logging.info(f"✅ Successfully loaded data for {ticker} (Features aligned)")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        error_msg = f"❌ Failed to load data for {ticker}: {str(e)}"
        errors.append((ticker, "Load Data", str(e)))
        logging.error(error_msg)
        console.print(f"[bold red]{error_msg}[/bold red]")
        return None, None, None, None


def tune_xgboost(X_train, y_train):
    """Tunes XGBoost hyperparameters using HalvingGridSearchCV."""
    try:
        console.print("[bold yellow]🔍 Tuning XGBoost hyperparameters...[/bold yellow]")
        logging.info("🔍 Starting hyperparameter tuning...")

        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 6, 9],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "gamma": [0, 0.1, 0.2],
            "reg_lambda": [0.1, 1, 10],
        }

        model = XGBRegressor(random_state=42, n_jobs=1)

        grid_search = HalvingGridSearchCV(
            model,
            param_grid,
            scoring="neg_mean_squared_error",
            cv=3,
            factor=2,
            n_jobs=1,
            verbose=0,
            error_score="raise",
        )

        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        console.print(
            f"[bold green]✅ Best XGBoost Hyperparameters: {best_params}[/bold green]"
        )
        logging.info(f"✅ Best XGBoost Hyperparameters: {best_params}")

        return best_params
    except Exception as e:
        error_msg = f"❌ Hyperparameter tuning failed: {str(e)}"
        errors.append(("Tuning", "XGBoost Hyperparameter Tuning", str(e)))
        logging.error(error_msg)
        console.print(f"[bold red]{error_msg}[/bold red]")
        return None


def train_best_xgboost(X_train, y_train, X_test, y_test, best_params):
    """Trains the best XGBoost model with early stopping, ensuring compatibility across versions."""
    try:
        console.print(
            "[bold yellow]🚀 Training best XGBoost model with early stopping...[/bold yellow]"
        )
        logging.info("🚀 Training best XGBoost model...")

        # ✅ Convert training data into DMatrix format (best for xgb.train)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # ✅ Extract `n_estimators` from best_params & remove it from params dict
        num_boost_round = best_params.pop("n_estimators", 200)

        # ✅ Use correct parameter name for boosting rounds
        model = xgb.train(
            params=best_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,  # ✅ Correct way to pass boosting rounds
            evals=[(dtest, "validation")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )

        logging.info("✅ XGBoost training completed successfully.")
        return model
    except Exception as e:
        error_msg = f"❌ Model training failed: {str(e)}"
        errors.append(("Training", "XGBoost Training", str(e)))
        logging.error(error_msg)
        console.print(f"[bold red]{error_msg}[/bold red]")
        return None


def train_and_save_xgboost(ticker):
    """Main function for training XGBoost on a ticker."""
    try:
        console.print(f"[bold yellow]⏳ Training XGBoost for {ticker}...[/bold yellow]")
        logging.info(f"⏳ Starting training for {ticker}...")

        X_train, y_train, X_test, y_test = load_data(ticker)
        if X_train is None:
            return

        best_params = tune_xgboost(X_train, y_train)
        if best_params is None:
            return

        best_model = train_best_xgboost(X_train, y_train, X_test, y_test, best_params)
        if best_model is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = os.path.join(MODEL_PATH, f"xgboost_{ticker}_{timestamp}.joblib")
        joblib.dump(best_model, model_file)

        logging.info(f"✅ XGBoost model saved: {model_file}")

    except Exception as e:
        error_msg = f"❌ Training & saving failed for {ticker}: {str(e)}"
        errors.append((ticker, "Train & Save", str(e)))
        logging.error(error_msg)
        console.print(f"[bold red]{error_msg}[/bold red]")


def process_all_tickers(num_tickers=None):
    """Trains XGBoost models for multiple tickers."""
    tickers = [
        f.split("_")[0]
        for f in os.listdir(PROCESSED_DATA_PATH)
        if f.endswith("_train.csv")
    ]

    if num_tickers is not None:
        tickers = tickers[:num_tickers]

    console.print(
        f"[bold cyan]🚀 Training XGBoost for {len(tickers)} tickers...[/bold cyan]"
    )

    with Pool(processes=min(len(tickers), cpu_count())) as pool:
        pool.map(train_and_save_xgboost, tickers)

    console.print("[bold green]✅ All XGBoost models trained and saved![/bold green]")

    if errors:
        console.print("\n[bold red]⚠ ERROR SUMMARY ⚠[/bold red]")
        table = Table(title="Errors Encountered")
        table.add_column("Ticker", justify="left", style="cyan")
        table.add_column("Stage", justify="left", style="yellow")
        table.add_column("Error", justify="left", style="red")

        for error in errors:
            table.add_row(*error)

        console.print(table)


if __name__ == "__main__":
    num_tickers = int(sys.argv[1]) if len(sys.argv) > 1 else None
    process_all_tickers(num_tickers)
