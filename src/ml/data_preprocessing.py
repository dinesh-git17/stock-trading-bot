import logging
import os
import warnings

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.theme import Theme
from sklearn.preprocessing import MinMaxScaler

from src.tools.utils import save_scaler, setup_logging  # ✅ Corrected import

# ✅ Suppress Warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ✅ Setup Logging
setup_logging("data/logs/data_preprocessing.log")

# ✅ Custom Theme for Console
custom_theme = Theme(
    {
        "success": "bold green",
        "info": "bold blue",
        "warning": "bold yellow",
        "error": "bold red",
    }
)
console = Console(theme=custom_theme)

logging.info("Starting data preprocessing...")

DATA_DIR = "data/training_data"
os.makedirs(DATA_DIR, exist_ok=True)  # ✅ Ensure directory exists


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the given DataFrame using MinMaxScaler.

    Args:
        df (pd.DataFrame): Data to normalize.

    Returns:
        pd.DataFrame: Normalized data.
    """
    logging.info("Applying MinMaxScaler normalization...")

    with Progress(
        TextColumn("[info]⏳ Processing:[/]"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Normalizing data...", total=len(df))

        # ✅ Convert NaN values to zero before scaling
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)  # ✅ Ensure no NaN values remain

        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(
            df.iloc[:, 2:]
        )  # Normalize numerical columns only
        save_scaler(scaler)  # ✅ Save scaler for future use

        progress.update(task, advance=len(df))

    logging.info("Normalization complete.")
    return pd.DataFrame(normalized_data, columns=df.columns[2:])


def preprocess_data(df: pd.DataFrame) -> None:
    """
    Apply preprocessing steps and save processed data as CSVs.

    Args:
        df (pd.DataFrame): Raw stock data.
    """
    logging.info("Starting preprocessing steps...")

    # ✅ Fix `FutureWarning`: Use `.ffill()` explicitly instead of `method="ffill"`
    df.iloc[:, 2:] = df.iloc[:, 2:].ffill()

    # ✅ Ensure integer columns remain integers
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype(int)

    df.iloc[:, 2:] = normalize_data(df)  # ✅ Normalize only numerical columns

    logging.info("Preprocessing complete. Storing CSVs...")

    # ✅ Store preprocessed CSVs per ticker
    for ticker, ticker_df in df.groupby("ticker"):
        filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
        ticker_df.to_csv(filepath, index=False)
        logging.info(f"✅ Preprocessed data stored for {ticker} -> {filepath}")

    console.print(f"[success]✅ Preprocessed data stored in {DATA_DIR}![/success]")
