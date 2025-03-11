import concurrent.futures
import logging
import os

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from scipy.stats import zscore

# ✅ Setup Logging
LOG_FILE = "data/logs/data_cleaning.log"
os.makedirs("data/logs", exist_ok=True)

# ✅ Insert 5 blank lines before logging new logs
with open(LOG_FILE, "a") as log_file:
    log_file.write("\n" * 5)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# ✅ Directories
PROCESSED_DATA_DIR = "data/processed/"
CLEANED_DATA_DIR = "data/cleaned/"
os.makedirs(CLEANED_DATA_DIR, exist_ok=True)  # Ensure cleaned data directory exists

# ✅ Parallel processing threads
THREADS = 4  # Adjust based on system resources


### **🚀 Handle Missing Values**
def handle_missing_values(df):
    """
    Handles missing values using forward fill and interpolation.
    """
    df.ffill(inplace=True)  # ✅ Forward fill missing values
    df.bfill(inplace=True)  # ✅ Backward fill as a second safeguard
    df.interpolate(method="linear", inplace=True)  # ✅ Linear interpolation
    return df


### **🚀 Remove Outliers**
def remove_outliers(df, column="close", threshold=3.0):
    """
    Removes outliers using Z-score method while avoiding SettingWithCopyWarning.
    """
    if column in df.columns:
        df = df.copy()  # ✅ Ensure we're working on a separate DataFrame
        df["zscore"] = zscore(df[column].dropna())
        df = df[df["zscore"].abs() < threshold].copy()  # ✅ Avoid modifying a slice
        df.drop(columns=["zscore"], inplace=True)  # ✅ Safe drop without warnings
    return df


### **🚀 Clean Stock Data**
def clean_stock_data(ticker):
    """
    Cleans stock data by handling missing values and removing outliers.
    Saves cleaned data to data/cleaned/.
    """
    processed_file = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")

    if not os.path.exists(processed_file):
        logging.warning(f"⚠ Processed data file missing for {ticker}. Skipping...")
        return

    df = pd.read_csv(processed_file)

    if df.empty:
        logging.warning(f"⚠ No data found in {processed_file}. Skipping...")
        return

    # ✅ Ensure correct date parsing
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df.set_index("Date", inplace=True)  # ✅ Use date as index
    else:
        logging.warning(f"⚠ No valid 'Date' column found in {ticker}. Skipping...")
        return

    # ✅ Ensure lowercase column names
    df.columns = df.columns.str.lower()

    # ✅ Handle missing values
    df = handle_missing_values(df)

    # ✅ Remove outliers
    df = remove_outliers(df, column="close")

    # ✅ Save cleaned data
    cleaned_file = os.path.join(CLEANED_DATA_DIR, f"{ticker}_cleaned.csv")
    df.to_csv(cleaned_file)
    logging.info(f"✅ Saved cleaned data for {ticker} to {cleaned_file}")


### **🚀 Parallel Processing for All Stocks**
def process_all_stocks():
    """
    Cleans all available stock data in data/processed/ using multi-threading.
    """
    files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith("_processed.csv")]
    tickers = [file.replace("_processed.csv", "") for file in files]

    console.print(
        f"\n[bold cyan]🚀 Cleaning data for {len(tickers)} stocks...[/bold cyan]\n"
    )

    with Progress(
        TextColumn("[bold yellow]Processing:[/bold yellow]"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing", total=len(tickers))

        with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = {
                executor.submit(clean_stock_data, ticker): ticker for ticker in tickers
            }

            for future in concurrent.futures.as_completed(futures):
                future.result()  # Ensure execution completes
                progress.update(task, advance=1)

    console.print(f"\n[bold green]✅ Done cleaning stock data![/bold green]\n")


### **🚀 Run the Script**
if __name__ == "__main__":
    process_all_stocks()
