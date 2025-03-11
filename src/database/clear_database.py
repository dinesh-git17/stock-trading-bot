import logging
import os

import psycopg2
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="data/logs/database_clear.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()

# Database Configuration
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")


def connect_db():
    """Connect to the PostgreSQL database."""
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


def drop_all_views(cursor):
    """Drops all views from the database."""
    try:
        console.print("[bold yellow]⚠️ Dropping all views...[/bold yellow]")
        cursor.execute(
            """
            DO $$ 
            DECLARE 
                r RECORD;
            BEGIN 
                FOR r IN (SELECT table_name FROM information_schema.views WHERE table_schema = 'public') LOOP 
                    EXECUTE 'DROP VIEW IF EXISTS ' || quote_ident(r.table_name) || ' CASCADE'; 
                END LOOP; 
            END $$;
            """
        )
        console.print(
            "[bold green]✅ All views have been dropped successfully![/bold green]"
        )
        logging.info("All views dropped successfully.")

    except Exception as e:
        console.print(f"[bold red]❌ Error dropping views:[/bold red] {e}")
        logging.error(f"Error dropping views: {e}")


def drop_all_tables(cursor):
    """Drops all tables from the database."""
    try:
        console.print("[bold yellow]⚠️ Dropping all tables...[/bold yellow]")
        cursor.execute(
            """
            DO $$ 
            DECLARE 
                r RECORD;
            BEGIN 
                FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP 
                    EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE'; 
                END LOOP; 
            END $$;
            """
        )
        console.print(
            "[bold green]✅ All tables have been dropped successfully![/bold green]"
        )
        logging.info("All tables dropped successfully.")

    except Exception as e:
        console.print(f"[bold red]❌ Error dropping tables:[/bold red] {e}")
        logging.error(f"Error dropping tables: {e}")


def clear_database():
    """Drops all views and tables to completely reset the database."""
    conn = connect_db()
    if not conn:
        return

    with conn.cursor() as cur:
        drop_all_views(cur)
        drop_all_tables(cur)

    conn.close()


if __name__ == "__main__":
    clear_database()
