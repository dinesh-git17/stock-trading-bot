import logging
import os
import pickle

from dotenv import load_dotenv
from sqlalchemy import create_engine


def setup_logging(log_file: str):
    """
    Setup logging for the application.

    Args:
        log_file (str): Path to the log file.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Additional log setup before actual logging begins
    with open(log_file, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("LOGGING SESSION STARTED\n")
        f.write("=" * 50 + "\n")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def get_database_engine():
    """
    Establish a PostgreSQL database connection using credentials from .env.

    Returns:
        sqlalchemy.engine.Engine: Database engine object.
    """
    load_dotenv()

    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(db_url)


def save_scaler(scaler, filepath="models/scaler.pkl"):
    """
    Saves the scaler to a file.

    Args:
        scaler (object): Scikit-learn scaler object.
        filepath (str): File path to save the scaler.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(filepath="models/scaler.pkl"):
    """
    Loads the scaler from a file.

    Args:
        filepath (str): File path of the scaler.

    Returns:
        object: Loaded scaler.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)
