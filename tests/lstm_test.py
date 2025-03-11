import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from prettytable import PrettyTable  # Install with: pip install prettytable
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
MODEL_PATH = "models/"
SCALER_PATH = "models/scalers/"
PROCESSED_DATA_PATH = "data/processed_data/"
SEQUENCE_LENGTH = 60  # Ensure it matches training config


# Function to load model
def load_model(ticker):
    model_file = os.path.join(MODEL_PATH, f"lstm_{ticker}.h5")
    if os.path.exists(model_file):
        return tf.keras.models.load_model(model_file)
    else:
        raise FileNotFoundError(f"Model file not found for {ticker}")


# Function to load and preprocess data
def load_test_data(ticker):
    """Loads and preprocesses test data, ensuring feature consistency."""
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")
    scaler_file = os.path.join(SCALER_PATH, f"{ticker}_scaler.pkl")

    if not os.path.exists(test_file) or not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Test data or scaler missing for {ticker}")

    df_test = pd.read_csv(test_file)

    # Load the scaler and retrieve the feature names used during training
    scaler = joblib.load(scaler_file)

    # Identify only the columns that were used for scaling
    training_features = scaler.feature_names_in_

    # Ensure the test data has the same columns and order
    df_test = df_test[training_features]

    # Extract actual target values (last column is assumed to be the target)
    y_test_actual = df_test.iloc[SEQUENCE_LENGTH:, -1].values

    # Scale the test data using the fitted scaler
    data_scaled = scaler.transform(df_test)

    # Prepare sequences
    X_test = []
    for i in range(SEQUENCE_LENGTH, len(data_scaled)):
        X_test.append(data_scaled[i - SEQUENCE_LENGTH : i])

    return np.array(X_test), y_test_actual, scaler


# Function to evaluate model
def evaluate_model(model, X_test, y_test_actual, scaler):
    y_pred_scaled = model.predict(X_test)

    # Inverse transform predictions
    y_pred = scaler.inverse_transform(
        np.hstack(
            [np.zeros((y_pred_scaled.shape[0], X_test.shape[2] - 1)), y_pred_scaled]
        )
    )[
        :, -1
    ]  # Extract only the last column (target)

    # Compute metrics
    mae = mean_absolute_error(y_test_actual, y_pred)
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, y_pred)

    # Determine performance quality
    performance = "Good" if r2 > 0.75 else "Needs Improvement"

    # Create DataFrame for results
    results_df = pd.DataFrame(
        {
            "Actual Price": y_test_actual,
            "Predicted Price": y_pred,
            "Performance": performance,
        }
    )

    return results_df, {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}


# Function to display results in a table (terminal)
def display_table(df):
    table = PrettyTable()
    table.field_names = ["Actual Price", "Predicted Price", "Performance"]

    for _, row in df.head(10).iterrows():  # Show first 10 rows
        table.add_row(row.tolist())

    print("\nPredicted vs Actual Prices:")
    print(table)


# Function to plot results
def plot_results(results_df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(results_df["Actual Price"], label="Actual Price", color="blue")
    plt.plot(
        results_df["Predicted Price"],
        label="Predicted Price",
        color="red",
        linestyle="dashed",
    )
    plt.title(f"Stock Price Prediction - {ticker}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


# Main execution
if __name__ == "__main__":
    ticker = "AAPL"  # Change to desired stock ticker

    try:
        print(f"Evaluating model for {ticker}...")

        # Load model and data
        model = load_model(ticker)
        X_test, y_test_actual, scaler = load_test_data(ticker)

        # Evaluate model
        results_df, metrics = evaluate_model(model, X_test, y_test_actual, scaler)

        # Display evaluation metrics
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Display table in terminal
        display_table(results_df)

        # Save results as a CSV file
        results_df.to_csv(f"{ticker}_evaluation_results.csv", index=False)
        print(f"\nResults saved as {ticker}_evaluation_results.csv")

        # Plot results
        plot_results(results_df, ticker)

    except FileNotFoundError as e:
        print(f"Error: {e}")
