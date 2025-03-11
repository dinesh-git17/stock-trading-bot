import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from tensorflow.keras.models import load_model

# ✅ Paths
MODEL_PATH = "models/"
SCALER_PATH = "models/scalers/"
PROCESSED_DATA_PATH = "data/processed_data/"
SEQUENCE_LENGTH = 60

console = Console()


def load_trained_model(ticker):
    """Loads a trained LSTM model for a specific stock ticker."""
    model_file = os.path.join(MODEL_PATH, f"lstm_{ticker}.h5")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found!")
    return load_model(model_file)


def load_scaler(ticker):
    """Loads the MinMaxScaler used for training."""
    scaler_file = os.path.join(SCALER_PATH, f"{ticker}_scaler.pkl")
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Scaler file {scaler_file} not found!")
    return joblib.load(scaler_file)


def load_test_data(ticker):
    """Loads the actual test data including the true stock prices."""
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test data file {test_file} not found!")

    df_test = pd.read_csv(test_file)
    return df_test


def preprocess_test_data(ticker):
    """Loads and preprocesses test data to match the training format."""
    df_test = load_test_data(ticker)
    df_test = df_test.select_dtypes(include=[np.number])

    # Load and apply the saved scaler
    scaler = load_scaler(ticker)
    expected_features = scaler.feature_names_in_

    for col in expected_features:
        if col not in df_test.columns:
            df_test[col] = 0  # Fill missing features with zeros

    df_test = df_test[expected_features]
    scaled_data = scaler.transform(df_test)

    # Create sequences
    X_test = []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X_test.append(scaled_data[i - SEQUENCE_LENGTH : i, :])

    return (
        np.array(X_test),
        df_test.iloc[SEQUENCE_LENGTH:]["close"].values,
    )  # Return true close prices too


def predict_stock_price(ticker):
    """Uses the trained LSTM model to predict stock prices."""
    model = load_trained_model(ticker)
    X_test, actual_prices = preprocess_test_data(ticker)

    if len(X_test) == 0:
        raise ValueError("Not enough test data to make predictions.")

    predictions = model.predict(X_test)
    return predictions, actual_prices


def inverse_transform_predictions(predictions, ticker):
    """Transforms predictions back to the original stock price scale."""
    scaler = load_scaler(ticker)

    # Ensure predictions are reshaped to match scaler expectations
    predictions = np.array(predictions).reshape(-1, 1)

    # Create a dummy array of zeros with the same feature count
    dummy_data = np.zeros((predictions.shape[0], len(scaler.feature_names_in_)))
    close_index = list(scaler.feature_names_in_).index("close")
    dummy_data[:, close_index] = predictions[:, 0]

    # Apply inverse transformation
    transformed_predictions = scaler.inverse_transform(dummy_data)[:, close_index]
    return transformed_predictions


def display_results_table(ticker, actual_prices, predicted_prices):
    """Displays a clear Rich table comparing actual vs. predicted prices, with prediction quality."""
    table = Table(title=f"{ticker} Stock Price Prediction vs Actual", show_lines=True)
    table.add_column("Index", justify="center", style="cyan", no_wrap=True)
    table.add_column("Actual Price ($)", justify="center", style="green")
    table.add_column("Predicted Price ($)", justify="center", style="blue")
    table.add_column("Difference ($)", justify="center", style="red")
    table.add_column("Prediction Quality", justify="center", style="magenta")

    for i in range(min(10, len(actual_prices))):
        diff = round(abs(actual_prices[i] - predicted_prices[i]), 2)
        percentage_error = (diff / actual_prices[i]) * 100

        # Determine prediction quality
        if percentage_error <= 1:
            quality = "✅ Good"
        elif percentage_error <= 3:
            quality = "⚠️ Moderate"
        else:
            quality = "❌ Poor"

        table.add_row(
            str(i + 1),
            f"{actual_prices[i]:.2f}",
            f"{predicted_prices[i]:.2f}",
            f"{diff:.2f}",
            quality,
        )

    console.print(table)
    console.print("\n💡 [bold yellow]Prediction Quality Explanation:[/bold yellow]")
    console.print(
        "✅ [bold green]Good[/bold green]: Difference ≤ 1% of actual price (Highly accurate)"
    )
    console.print(
        "⚠️ [bold yellow]Moderate[/bold yellow]: Difference between 1% - 3% (Acceptable)"
    )
    console.print("❌ [bold red]Poor[/bold red]: Difference > 3% (Inaccurate)\n")


# ✅ Example Usage
ticker = "AAPL"  # Replace with your stock ticker
console.print(
    f"\n[bold cyan]📊 Running stock price predictions for {ticker}...[/bold cyan]\n"
)

predictions, actual_prices = predict_stock_price(ticker)
original_predictions = inverse_transform_predictions(predictions, ticker)

console.print(
    f"\n✅ [bold green]Predictions completed! Displaying results for {ticker}.[/bold green]\n"
)

# ✅ Show Rich Table
display_results_table(ticker, actual_prices, original_predictions)

# ✅ Plot Predictions vs Actual Prices
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label="Actual Prices", color="green", linestyle="dashed")
plt.plot(original_predictions, label="Predicted Prices", color="blue")
plt.title(f"{ticker} Stock Price Prediction vs Actual")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
