import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from rich.console import Console
from rich.table import Table
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Paths
PROCESSED_DATA_PATH = "data/processed_data/"
MODEL_PATH = "models/"

console = Console()


def load_test_data(ticker):
    """Loads test data for a given ticker."""
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")

    if not os.path.exists(test_file):
        console.print(f"[bold red]❌ No test data found for {ticker}.[/bold red]")
        return None, None

    df_test = pd.read_csv(test_file).select_dtypes(include=[np.number])

    # ✅ Ensure "close" column is present
    if "close" not in df_test.columns:
        console.print(
            f"[bold red]❌ 'close' column missing in test data for {ticker}.[/bold red]"
        )
        return None, None

    X_test, y_test = df_test.drop(columns=["close"]), df_test["close"]

    return X_test, y_test


def load_model(ticker):
    """Loads the trained XGBoost model for a given ticker."""
    model_files = [
        f
        for f in os.listdir(MODEL_PATH)
        if f.startswith(f"xgboost_{ticker}_") and f.endswith(".joblib")
    ]

    if not model_files:
        console.print(f"[bold red]❌ No trained model found for {ticker}.[/bold red]")
        return None

    # ✅ Load the most recent model
    model_files.sort(reverse=True)  # Sort by latest timestamp
    latest_model_file = os.path.join(MODEL_PATH, model_files[0])

    console.print(f"[bold cyan]📥 Loading model: {latest_model_file}[/bold cyan]")
    model = joblib.load(latest_model_file)

    return model


def evaluate_model(model, X_test, y_test, ticker):
    """Evaluates the trained model and displays a comparison table."""
    try:
        # ✅ Ensure test features are in the same order as training features
        if isinstance(model, xgb.Booster):
            train_feature_names = model.feature_names
        else:
            train_feature_names = model.get_booster().feature_names

        X_test = X_test[train_feature_names]

        # ✅ Convert X_test to DMatrix if model was trained using `xgb.train()`
        X_test_dmatrix = (
            xgb.DMatrix(X_test) if isinstance(model, xgb.Booster) else X_test
        )

        # ✅ Predict
        y_pred = model.predict(X_test_dmatrix)

        # ✅ Compute Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        console.print(f"[bold green]✅ Model Evaluation Metrics:[/bold green]")
        console.print(f"📊 RMSE: {rmse:.4f}")
        console.print(f"📊 MAE: {mae:.4f}")
        console.print(f"📊 R² Score: {r2:.4f}")

        # ✅ Model Quality Indicator
        if r2 > 0.8:
            model_quality = "[bold green]✅ Excellent Model[/bold green]"
        elif 0.5 <= r2 <= 0.8:
            model_quality = "[bold yellow]⚠️ Moderate Model[/bold yellow]"
        else:
            model_quality = "[bold red]❌ Weak Model[/bold red]"

        console.print(f"📈 Model Performance: {model_quality}")

        # ✅ Display comparison table
        table = Table(title=f"📈 Actual vs Predicted for {ticker}")
        table.add_column("Index", justify="center", style="cyan")
        table.add_column("Actual Close", justify="center", style="yellow")
        table.add_column("Predicted Close", justify="center", style="green")
        table.add_column("Difference", justify="center", style="red")

        for i in range(min(20, len(y_test))):  # ✅ Show 20 rows instead of 10
            actual = y_test.iloc[i]
            predicted = y_pred[i]
            diff = abs(actual - predicted)
            table.add_row(str(i), f"{actual:.2f}", f"{predicted:.2f}", f"{diff:.2f}")

        console.print(table)

        # ✅ Plot Actual vs Predicted
        plot_predictions(y_test, y_pred, ticker)

    except Exception as e:
        console.print(f"[bold red]❌ Error in evaluation: {str(e)}[/bold red]")


def plot_predictions(y_test, y_pred, ticker):
    """Plots Actual vs Predicted values."""
    plt.figure(figsize=(10, 5))
    plt.plot(
        y_test.values, label="Actual Prices", marker="o", linestyle="-", color="blue"
    )
    plt.plot(y_pred, label="Predicted Prices", marker="x", linestyle="--", color="red")

    plt.title(f"Actual vs Predicted Prices for {ticker}")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    ticker = input("Enter the stock ticker (e.g., AAPL, TSLA, etc.): ").strip().upper()

    # ✅ Load test data
    X_test, y_test = load_test_data(ticker)
    if X_test is None:
        exit(1)

    # ✅ Load trained model
    model = load_model(ticker)
    if model is None:
        exit(1)

    # ✅ Evaluate and display results
    evaluate_model(model, X_test, y_test, ticker)
