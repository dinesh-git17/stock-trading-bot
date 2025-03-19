import logging
import os
import pickle
import sys
import traceback
from pathlib import Path
from shutil import get_terminal_size
from threading import Lock

from dotenv import load_dotenv
from rich.box import HEAVY
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sqlalchemy import create_engine

# Thread lock for thread-safe logging setup
_log_lock = Lock()

# Initialize Rich Console for error display
console = Console()


def handle_exceptions(func):
    """
    A decorator to catch and display unexpected errors in a thick-box styled error panel.
    It suppresses only error messages (stderr) but allows normal console output (stdout).

    Args:
        func (function): The function to wrap.

    Returns:
        function: Wrapped function with error handling.
    """

    def wrapper(*args, **kwargs):
        # Suppress only stderr before execution
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")

        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Restore stderr before displaying error
            sys.stderr = original_stderr

            # Capture detailed traceback information
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)

            # Extract key error details
            tb_last = traceback.extract_tb(exc_traceback)[-1]
            error_file = tb_last.filename
            error_line = tb_last.lineno
            error_function = tb_last.name
            error_message = str(e)

            # Get Terminal Size for Proper Box Width
            terminal_width = get_terminal_size((120, 40)).columns
            panel_width = min(100, terminal_width - 10)  # Keeps it within screen width

            # Limit traceback lines to avoid terminal overflow
            max_traceback_lines = 6  # Reduce lines to prevent excessive scrolling
            tb_lines = tb_lines[:max_traceback_lines]
            if len(tb_lines) == max_traceback_lines:
                tb_lines.append("... (truncated)")

            # Convert traceback to red text
            traceback_text = f"```diff\n- {''.join(tb_lines)}```"

            # Format error output using Markdown with proper alignment
            error_markdown = f"""
## 🛑 ERROR REPORT

⚠️ **An unexpected exception occurred!**

**✘ Error:** `{error_message}`

**📄 File:** `{error_file}`

**📍 Line:** `{error_line}`

**🔹 Function:** `{error_function}`

**🔎 Traceback:**

{traceback_text}
            """

            # Display formatted error in a Rich panel with Markdown and red traceback
            console.print(
                Panel(
                    Markdown(error_markdown),
                    border_style="bright_red",
                    padding=(1, 4),  # Ensure even spacing
                    width=panel_width,  # Ensures proper sizing
                    expand=False,
                    box=HEAVY,  # Thick box border
                )
            )

    return wrapper


@handle_exceptions
def setup_logging(log_file: str, level=logging.INFO):
    """
    Sets up logging for the application in a robust and configurable way.

    Args:
        log_file (str): Path to the log file.
        level (int): Logging level (default is logging.INFO).
    """
    with _log_lock:  # Ensures thread safety
        log_path = Path(log_file)

        # Ensure the directory exists before creating the log file
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Failed to create log directory: {e}")

        # Check if logging is already configured to prevent duplicate logs
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                filename=log_file,
                level=level,
                format="%(asctime)s - %(levelname)s - %(message)s",
                encoding="utf-8",
            )

            # Additional log setup before actual logging begins
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 50 + "\n")
                f.write("LOGGING SESSION STARTED\n")
                f.write("=" * 50 + "\n")

        logging.info("✅ Logging successfully initialized!")


@handle_exceptions
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


@handle_exceptions
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


@handle_exceptions
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
