import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
import logging
from rich.console import Console
import tensorflow as tf
import psycopg2
from rich.progress import Progress
from tensorflow.keras.layers import Input
from rich.progress import Progress, BarColumn, TextColumn

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/lstm_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# Database connection configuration
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")


# Database connection function
def connect_db():
    """Connects to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        return conn
    except Exception as e:
        console.print(f"[bold red]❌ Database connection error:[/bold red] {e}")
        logging.error(f"Database connection error: {e}")
        return None


def load_data(ticker):
    """Loads preprocessed stock data."""
    file_path = f"data/processed_data/{ticker}_processed.csv"
    if not os.path.exists(file_path):
        logging.warning(f"Processed data file for {ticker} does not exist!")
        return None

    df = pd.read_csv(file_path)
    return df


def prepare_data(df, ticker, sequence_length=60):
    """Prepares the data for LSTM model."""
    # Drop the 'date' column as it's not needed for the model
    df = df.drop(columns=["date"])

    # Use MinMaxScaler only on the 'close' column (or the column you're predicting)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Only scale the 'close' column (assuming you're predicting close prices)
    close_data = df[["close"]]  # Extract 'close' column
    scaled_data = scaler.fit_transform(close_data.values)

    # Check if the dataset has enough data
    if len(scaled_data) < sequence_length:
        logging.warning(
            f"Dataset for {ticker} is too small with only {len(scaled_data)} rows. Skipping this ticker."
        )
        return None  # Skip this ticker

    # Prepare the sequences for LSTM
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(
            scaled_data[i - sequence_length : i, 0]
        )  # Last `sequence_length` closing prices
        y.append(scaled_data[i, 0])  # Next day's closing price

    # If there are no sequences created, return an error or empty list
    if len(X) == 0:
        raise ValueError(
            "Not enough data to create sequences. Please ensure your dataset is sufficiently large."
        )

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Ensure X has the correct shape for LSTM (samples, time_steps, features)
    if X.ndim == 2:
        X = np.reshape(
            X, (X.shape[0], X.shape[1], 1)
        )  # Reshaping to 3D for LSTM input (samples, time_steps, features)

    return X, y, scaler


def build_lstm_model(input_shape):
    """Builds the LSTM model."""
    model = Sequential()

    # Add Input layer as the first layer
    model.add(Input(shape=input_shape))

    # First LSTM layer with 50 units
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=1))  # Output layer

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(ticker):
    """Trains the LSTM model."""
    console.print(f"\n[bold blue]Training model for {ticker}...[/bold blue]")

    # Load the preprocessed data
    df = load_data(ticker)
    if df is None:
        return

    # Prepare the data for LSTM
    result = prepare_data(df, ticker)  # Pass the ticker to prepare_data
    if result is None:
        logging.warning(f"Skipping {ticker} due to insufficient data.")
        return  # Skip this ticker if the data preparation failed

    X, y, scaler = result

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Build and train the LSTM model
    model = build_lstm_model((X_train.shape[1], 1))

    with Progress(
        TextColumn("[bold yellow]Epoch:[bold yellow] {task.completed} / {task.total}"),
        BarColumn(),
        console=console,
    ) as progress:
        # Add a task to track progress for training epochs
        task = progress.add_task(f"Training {ticker}", total=10)

        # Training the model and updating progress
        for epoch in range(10):
            model.fit(
                X_train,
                y_train,
                epochs=1,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0,
            )
            progress.update(task, advance=1)
            console.print(
                f"[bold green]Epoch {epoch + 1} completed for {ticker}[/bold green]"
            )

    # Save the trained model
    model.save(f"models/lstm_model_{ticker}.h5")
    logging.info(f"Trained LSTM model for {ticker} saved successfully.")

    # Evaluate the model performance
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(
        predictions
    )  # Rescale back to original values
    y_test_rescaled = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )  # Rescale back to original values

    # Calculate RMSE (Root Mean Squared Error) for evaluation
    rmse = np.sqrt(np.mean(np.square(predictions - y_test_rescaled)))
    logging.info(f"RMSE for {ticker} model: {rmse}")

    # Display a clean result in the console
    console.print(f"[bold blue]Model trained and saved for {ticker}![/bold blue]")


def get_tickers_from_db():
    """Fetch all tickers from the database."""
    conn = connect_db()
    if not conn:
        return []
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT ticker FROM stocks;")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()

    return tickers

# Main loop to process all tickers
def train_all_models():
    tickers = ["AAPL", "TSLA", "GOOGL", "AMCR", "PLTR", "WMT"]  # Example tickers, could be fetched from DB
    for ticker in tickers:
        train_model(ticker)

if __name__ == "__main__":
    train_all_models()