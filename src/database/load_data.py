import logging
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/load_data.log"
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

# ✅ Database Configuration
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# ✅ Data Directories
CLEANED_DATA_DIR = "data/cleaned/"


### **🚀 Connect to PostgreSQL Database**
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
        conn.autocommit = True
        return conn
    except Exception as e:
        console.print(f"[bold red]❌ Database connection error:[/bold red] {e}")
        logging.error(f"Database connection error: {e}")
        return None


### **🚀 Insert Data into Database**
def insert_data(cursor, ticker, df):
    """Inserts cleaned stock data into `stocks` and `technical_indicators` tables."""
    try:
        stock_columns = [
            "ticker",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adjusted_close",
        ]
        indicator_columns = [
            "ticker",
            "date",
            "sma_50",
            "sma_200",
            "ema_50",
            "ema_200",
            "rsi_14",
            "adx_14",
            "atr_14",
            "cci_20",
            "williamsr_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "stoch_k",
            "stoch_d",
        ]

        # ✅ Ensure required columns exist
        df["ticker"] = ticker
        df.rename(columns={"Date": "date"}, inplace=True)

        # ✅ Insert into `stocks` table
        stock_values = df[stock_columns].values.tolist()
        stock_insert_query = f"""
            INSERT INTO stocks ({", ".join(stock_columns)}) 
            VALUES ({", ".join(["%s"] * len(stock_columns))})
            ON CONFLICT (ticker, date) DO NOTHING;
        """
        cursor.executemany(stock_insert_query, stock_values)

        # ✅ Insert into `technical_indicators` table
        indicator_values = df[indicator_columns].values.tolist()
        indicator_insert_query = f"""
            INSERT INTO technical_indicators ({", ".join(indicator_columns)}) 
            VALUES ({", ".join(["%s"] * len(indicator_columns))})
            ON CONFLICT (ticker, date) DO NOTHING;
        """
        cursor.executemany(indicator_insert_query, indicator_values)

        logging.info(f"✅ Successfully loaded data for {ticker}")

    except Exception as e:
        console.print(f"[bold red]❌ Error inserting data for {ticker}:[/bold red] {e}")
        logging.error(f"Error inserting data for {ticker}: {e}")


### **🚀 Process a Single Stock**
def process_stock(ticker, cursor):
    """Loads cleaned stock data for a given ticker and inserts it into the database."""
    cleaned_file = os.path.join(CLEANED_DATA_DIR, f"{ticker}_cleaned.csv")

    if not os.path.exists(cleaned_file):
        logging.warning(f"⚠ Cleaned data file missing for {ticker}")
        return

    df = pd.read_csv(cleaned_file, parse_dates=["Date"])

    if df.empty:
        logging.warning(f"⚠ No data found in {cleaned_file}")
        return

    insert_data(cursor, ticker, df)


### **🚀 Process All Stocks**
def process_all_stocks():
    """Loads all cleaned stock data into the database."""
    conn = connect_db()
    if not conn:
        return

    with conn.cursor() as cursor:
        files = [f for f in os.listdir(CLEANED_DATA_DIR) if f.endswith("_cleaned.csv")]
        tickers = [file.replace("_cleaned.csv", "") for file in files]

        console.print(f"\n🚀 Loading cleaned stock data for {len(tickers)} stocks...\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]Inserting Data into Database...[/bold yellow]"),
            console=console,
        ) as progress:
            task = progress.add_task("loading", total=len(tickers))

            for ticker in tickers:
                process_stock(ticker, cursor)
                progress.update(task, advance=1)

    conn.close()
    console.print(
        "[bold green]✅ All cleaned stock data loaded successfully![/bold green]"
    )


### **🚀 Run the Script**
if __name__ == "__main__":
    process_all_stocks()
