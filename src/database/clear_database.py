import logging
import os

from dotenv import load_dotenv
from rich.console import Console
from sqlalchemy import text

from src.tools.utils import get_database_engine, handle_exceptions, setup_logging

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/database_clear.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)

# ✅ Setup Console
console = Console()


@handle_exceptions
def drop_views(engine):
    """Drops all views from the database."""
    query = text(
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

    with engine.begin() as connection:
        connection.execute(query)

    console.print("[bold green]✅ All views dropped successfully![/bold green]")
    logger.info("All views dropped successfully.")


@handle_exceptions
def drop_tables(engine):
    """Drops all tables from the database."""
    query = text(
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

    with engine.begin() as connection:
        connection.execute(query)

    console.print("[bold green]✅ All tables dropped successfully![/bold green]")
    logger.info("All tables dropped successfully.")


def clear_database():
    """Drops all views and tables to completely reset the database."""
    engine = get_database_engine()
    if not engine:
        return

    drop_views(engine)
    drop_tables(engine)


if __name__ == "__main__":
    clear_database()
