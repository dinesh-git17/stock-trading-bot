import logging
import os

import psycopg2
from dotenv import load_dotenv
from rich.console import Console

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/database_setup.log"
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


### **🚀 Update `stock_info` Table**
def update_stock_info_table(cursor):
    """Ensures `stock_info` table has all required columns."""
    console.print(
        "[bold yellow]🔄 Checking and updating `stock_info` table...[/bold yellow]"
    )

    alter_queries = [
        "ALTER TABLE stock_info ADD COLUMN IF NOT EXISTS market_cap BIGINT;",
        "ALTER TABLE stock_info ADD COLUMN IF NOT EXISTS pe_ratio NUMERIC;",
        "ALTER TABLE stock_info ADD COLUMN IF NOT EXISTS eps NUMERIC;",
        "ALTER TABLE stock_info ADD COLUMN IF NOT EXISTS earnings_date DATE;",
    ]

    for query in alter_queries:
        cursor.execute(query)

    console.print(
        "[bold green]✅ `stock_info` table updated successfully![/bold green]"
    )
    logging.info("Updated `stock_info` table to include missing columns.")


### **🚀 Update `technical_indicators` Table**
def update_technical_indicators_table(cursor):
    """Ensures `technical_indicators` table matches the latest schema."""
    console.print(
        "[bold yellow]🔄 Checking and updating `technical_indicators` table...[/bold yellow]"
    )

    # ✅ Define the schema to support cleaned data structure
    required_columns = {
        "id": "SERIAL PRIMARY KEY",
        "ticker": "VARCHAR(10) NOT NULL",
        "date": "TIMESTAMP NOT NULL",
        "sma_50": "NUMERIC",
        "sma_200": "NUMERIC",
        "ema_50": "NUMERIC",
        "ema_200": "NUMERIC",
        "rsi_14": "NUMERIC",
        "adx_14": "NUMERIC",
        "atr_14": "NUMERIC",
        "cci_20": "NUMERIC",
        "williamsr_14": "NUMERIC",
        "macd": "NUMERIC",
        "macd_signal": "NUMERIC",
        "macd_hist": "NUMERIC",
        "bb_upper": "NUMERIC",
        "bb_middle": "NUMERIC",
        "bb_lower": "NUMERIC",
        "stoch_k": "NUMERIC",
        "stoch_d": "NUMERIC",
    }

    # ✅ Fetch existing columns
    cursor.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'technical_indicators';"
    )
    existing_columns = {row[0] for row in cursor.fetchall()}

    # ✅ Add missing columns dynamically
    for column, column_type in required_columns.items():
        if column not in existing_columns:
            cursor.execute(
                f"ALTER TABLE technical_indicators ADD COLUMN {column} {column_type};"
            )
            console.print(f"[bold green]➕ Added column:[/bold green] {column}")

    # ✅ Remove extra columns dynamically
    for column in existing_columns:
        if column not in required_columns:
            cursor.execute(
                f"ALTER TABLE technical_indicators DROP COLUMN {column} CASCADE;"
            )
            console.print(f"[bold red]❌ Removed extra column:[/bold red] {column}")

    console.print(
        "[bold green]✅ `technical_indicators` table is up to date![/bold green]"
    )
    logging.info("Updated `technical_indicators` table successfully.")


### **🚀 Create Tables**
def create_tables():
    """Creates all necessary tables in the PostgreSQL database."""
    conn = connect_db()
    if not conn:
        return

    with conn.cursor() as cur:
        try:
            # ✅ Create `stocks` table (OHLCV data)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS stocks (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    date TIMESTAMP NOT NULL,
                    open NUMERIC,
                    high NUMERIC,
                    low NUMERIC,
                    close NUMERIC,
                    volume BIGINT,
                    adjusted_close NUMERIC,
                    UNIQUE (ticker, date)
                );
            """
            )

            # ✅ Create `technical_indicators` table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    date TIMESTAMP NOT NULL,
                    sma_50 NUMERIC,
                    sma_200 NUMERIC,
                    ema_50 NUMERIC,
                    ema_200 NUMERIC,
                    rsi_14 NUMERIC,
                    adx_14 NUMERIC,
                    atr_14 NUMERIC,
                    cci_20 NUMERIC,
                    williamsr_14 NUMERIC,
                    macd NUMERIC,
                    macd_signal NUMERIC,
                    macd_hist NUMERIC,
                    bb_upper NUMERIC,
                    bb_middle NUMERIC,
                    bb_lower NUMERIC,
                    stoch_k NUMERIC,
                    stoch_d NUMERIC,
                    FOREIGN KEY (ticker, date) REFERENCES stocks (ticker, date) ON DELETE CASCADE,
                    UNIQUE (ticker, date)
                );
            """
            )

            # ✅ Create `stock_info` table (stores metadata like sector)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_info (
                    ticker VARCHAR(10) PRIMARY KEY,
                    company_name TEXT,
                    sector TEXT,
                    market_cap BIGINT,
                    pe_ratio NUMERIC,
                    eps NUMERIC,
                    earnings_date DATE
                );
            """
            )

            # ✅ Create `news_sentiment` table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    title TEXT,
                    sentiment_score NUMERIC,
                    UNIQUE (ticker, published_at)
                );
            """
            )

            update_stock_info_table(cur)
            update_technical_indicators_table(cur)

            console.print(
                "[bold green]✅ Database tables created successfully![/bold green]"
            )
            logging.info("Database tables created successfully.")

        except Exception as e:
            console.print(f"[bold red]❌ Error creating tables:[/bold red] {e}")
            logging.error(f"Error creating tables: {e}")

    conn.close()


### **🚀 Run the Script**
if __name__ == "__main__":
    create_tables()
