import numpy as np
import pandas as pd
import os
import logging
import warnings
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Conv1D,
    Flatten,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from multiprocessing import Pool, cpu_count, Manager
import joblib
import signal
import sys

# ✅ Fix multiprocessing issue on macOS
if sys.platform == "darwin":
    from multiprocessing import set_start_method

    set_start_method("fork")

# ✅ Load environment variables
load_dotenv()

# ✅ Setup logging (logs both to file and console)
LOG_FILE = "data/logs/stock_gpt_training.log"
os.makedirs("data/logs", exist_ok=True)

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
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(SCALER_PATH, exist_ok=True)

# ✅ Transformer Configuration
SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 50
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
DROPOUT_RATE = 0.2


# ✅ Handle Ctrl+C safely
def signal_handler(sig, frame):
    console.print("\n[bold red]❌ Process interrupted. Exiting safely...[/bold red]")
    logging.info("Process interrupted by user.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def count_saved_models():
    """Counts and logs the total number of trained models."""
    num_models = len([f for f in os.listdir(MODEL_PATH) if f.endswith(".keras")])
    console.print(
        f"[bold cyan]📊 Total StockGPT models saved: {num_models}[/bold cyan]"
    )
    logging.info(f"Total StockGPT models saved: {num_models}")
    return num_models


def load_data(ticker):
    """Loads preprocessed stock data for training/testing."""
    train_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_train.csv")
    test_file = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_test.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        logging.warning(f"No data found for {ticker}. Skipping training.")
        return None, None

    return pd.read_csv(train_file), pd.read_csv(test_file)


def prepare_data(df_train, df_test, ticker):
    """Prepares stock data for Transformer training."""
    df_train = df_train.select_dtypes(include=[np.number])
    df_test = df_test.select_dtypes(include=[np.number])

    feature_columns = df_train.columns.tolist()
    scaler_file = os.path.join(SCALER_PATH, f"{ticker}_scaler.pkl")

    scaler = MinMaxScaler()
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
    else:
        scaler.fit(df_train[feature_columns])
        joblib.dump(scaler, scaler_file)

    df_train, df_test = scaler.transform(df_train[feature_columns]), scaler.transform(
        df_test[feature_columns]
    )

    def create_sequences(data):
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(data)):
            X.append(data[i - SEQUENCE_LENGTH : i, :])
            y.append(data[i, feature_columns.index("close")])
        return np.array(X), np.array(y).reshape(-1, 1)

    X_train, y_train = create_sequences(df_train)
    X_test, y_test = create_sequences(df_test)

    # ✅ Fix: Check if `X_train` or `y_train` is empty and log a warning
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        logging.warning(f"{ticker} - Not enough data for training. Skipping...")
        console.print(f"[bold red]⚠ {ticker} - Not enough data. Skipping...[/bold red]")
        return None, None, None, None, None, None

    return X_train, y_train, X_test, y_test, scaler, feature_columns


def transformer_encoder(inputs):
    """Defines a single Transformer encoder block."""
    attn_output = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=D_MODEL)(
        inputs, inputs
    )
    attn_output = Dropout(DROPOUT_RATE)(attn_output)
    attn_output = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = Dense(FF_DIM, activation="relu")(attn_output)
    ffn_output = Dense(D_MODEL)(ffn_output)
    ffn_output = Dropout(DROPOUT_RATE)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(attn_output + ffn_output)


def build_transformer_model(input_shape):
    """Builds and compiles a Transformer model for stock price prediction."""
    inputs = Input(shape=input_shape)
    x = Dense(D_MODEL, activation="relu")(inputs)
    x = transformer_encoder(x)
    x = Conv1D(filters=32, kernel_size=3, activation="relu")(x)
    x = Flatten()(x)
    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(args):
    """Loads data, trains Transformer model, and saves best model."""
    ticker, completed_jobs, log_queue = args

    try:
        log_queue.put(f"{ticker:<6} ⏳ Training model...")
        logging.info(f"{ticker} - Training started.")

        df_train, df_test = load_data(ticker)
        if df_train is None or df_test is None:
            completed_jobs.append(ticker)
            return ticker, "❌ No Data"

        X_train, y_train, X_test, y_test, _, _ = prepare_data(df_train, df_test, ticker)
        if X_train is None:
            completed_jobs.append(ticker)
            return ticker, "❌ Insufficient Data"

        model = build_transformer_model((SEQUENCE_LENGTH, X_train.shape[2]))
        model_path = os.path.join(MODEL_PATH, f"stock_gpt_{ticker}.keras")

        model_checkpoint = ModelCheckpoint(
            model_path, save_best_only=True, monitor="val_loss", mode="min", verbose=0
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=0
        )

        model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[early_stopping, model_checkpoint],
        )

        model.save(model_path, save_format="keras")

        completed_jobs.append(ticker)
        log_queue.put(f"{ticker:<6} ✅ Model Trained & Saved")
        logging.info(f"{ticker} - Model trained & saved successfully.")
        return ticker, "✅ Model Trained & Saved"

    except Exception as e:
        completed_jobs.append(ticker)
        log_queue.put(f"{ticker:<6} ❌ Training Failed: {str(e)}")
        logging.error(f"{ticker} - Training Failed: {str(e)}")
        return ticker, f"❌ Training Failed: {str(e)}"


def process_all_models():
    """Trains all stock models in parallel and logs results."""
    tickers = [
        f.split("_")[0]
        for f in os.listdir(PROCESSED_DATA_PATH)
        if f.endswith("_train.csv")
    ]

    console.print(
        f"[bold cyan]🚀 Training StockGPT models for {len(tickers)} tickers...[/bold cyan]"
    )

    manager = Manager()
    completed_jobs = manager.list()
    log_queue = manager.Queue()

    with Pool(cpu_count()) as pool:
        pool.map(
            train_model, [(ticker, completed_jobs, log_queue) for ticker in tickers]
        )

    count_saved_models()


if __name__ == "__main__":
    process_all_models()
