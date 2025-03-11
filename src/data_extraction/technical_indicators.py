import concurrent.futures
import logging
import os

import pandas as pd
import pandas_ta as ta
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

# ✅ Setup Logging
LOG_FILE = "data/logs/technical_indicators.log"
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
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)  # Ensure processed data directory exists

# ✅ Max workers for parallel processing
THREADS = 4  # Adjust based on system resources

# ✅ Required columns for processing
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume", "adjusted_close"]


### **🚀 Read and Validate CSV**
def read_stock_data(ticker):
    """Reads stock data and ensures it has the necessary structure."""
    raw_file = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")

    if not os.path.exists(raw_file):
        logging.warning(f"⚠ Raw data file missing for {ticker}. Skipping...")
        return None

    df = pd.read_csv(raw_file)

    # ✅ Ensure proper date parsing
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df.set_index("Date", inplace=True)  # ✅ Use date as index
    else:
        logging.warning(f"⚠ No valid 'Date' column found in {ticker}.csv. Skipping...")
        return None

    # ✅ Ensure correct column names (case sensitivity fix)
    df.columns = df.columns.str.lower()

    # ✅ Ensure required columns exist
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        logging.warning(
            f"⚠ Missing columns {missing_cols} in {ticker}.csv. Skipping..."
        )
        return None

    return df


### **🚀 Compute Technical Indicators**
def compute_technical_indicators(ticker):
    """
    Compute multiple technical indicators for a given stock ticker.
    Saves processed data into data/processed/.
    """
    df = read_stock_data(ticker)
    if df is None:
        return  # Skip processing if file is invalid

    # ✅ Compute Technical Indicators
    df["SMA_50"] = ta.sma(df["close"], length=50)
    df["SMA_200"] = ta.sma(df["close"], length=200)
    df["EMA_50"] = ta.ema(df["close"], length=50)
    df["EMA_200"] = ta.ema(df["close"], length=200)
    df["RSI_14"] = ta.rsi(df["close"], length=14)
    df["ADX_14"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["ATR_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["CCI_20"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    df["WilliamsR_14"] = ta.willr(df["high"], df["low"], df["close"], length=14)

    # ✅ Compute MACD & Signal Line
    macd = ta.macd(df["close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_Signal"] = macd["MACDs_12_26_9"]
    df["MACD_Hist"] = macd["MACDh_12_26_9"]

    # ✅ Compute Bollinger Bands
    bbands = ta.bbands(df["close"], length=20)
    df["BB_Upper"] = bbands["BBU_20_2.0"]
    df["BB_Middle"] = bbands["BBM_20_2.0"]
    df["BB_Lower"] = bbands["BBL_20_2.0"]

    # ✅ Compute Stochastic Oscillator
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    df["Stoch_K"] = stoch["STOCHk_14_3_3"]
    df["Stoch_D"] = stoch["STOCHd_14_3_3"]

    # ✅ Drop NaN values after indicator calculations
    df.dropna(inplace=True)

    # ✅ Save processed data
    processed_file = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
    df.to_csv(processed_file)
    logging.info(
        f"✅ Processed technical indicators for {ticker} saved to {processed_file}"
    )


### **🚀 Parallel Processing for All Stocks**
def process_all_stocks():
    """
    Compute technical indicators for all available stock data in data/raw/ using multi-threading.
    """
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    tickers = [file.replace(".csv", "") for file in files]

    console.print(
        f"\n[bold cyan]🚀 Computing technical indicators for {len(tickers)} stocks...[/bold cyan]\n"
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
                executor.submit(compute_technical_indicators, ticker): ticker
                for ticker in tickers
            }

            for future in concurrent.futures.as_completed(futures):
                future.result()  # Ensure execution completes
                progress.update(task, advance=1)

    console.print(
        f"\n[bold green]✅ Done computing technical indicators![/bold green]\n"
    )


### **🚀 Run the Script**
if __name__ == "__main__":
    process_all_stocks()
