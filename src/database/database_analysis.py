import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from sqlalchemy import text

from src.tools.utils import get_database_engine, handle_exceptions, setup_logging

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
LOG_FILE = "data/logs/database_analysis.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)

# ✅ Setup Console
console = Console()

# ✅ Output File
REPORT_FILE = "data/database_report.md"


@handle_exceptions
def get_tables(engine):
    """Fetches all tables from the database."""
    query = text(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
    )
    with engine.connect() as conn:
        return [row[0] for row in conn.execute(query)]


@handle_exceptions
def get_table_schema(engine, table_name):
    """Fetches schema details of a table."""
    query = text(
        """
        SELECT column_name, data_type, is_nullable, COALESCE(column_default::text, 'NULL') 
        FROM information_schema.columns 
        WHERE table_name = :table_name;
    """
    )
    with engine.connect() as conn:
        return conn.execute(query, {"table_name": table_name}).fetchall()


@handle_exceptions
def get_sample_data(engine, table_name):
    """Fetches the first 5 rows of a table."""
    query = text(f"SELECT * FROM {table_name} LIMIT 5;")
    with engine.connect() as conn:
        return conn.execute(query).fetchall()


@handle_exceptions
def get_date_range(engine, table_name):
    """Finds the date range of stock price data in a table."""
    query = text(
        """
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = :table_name 
        AND data_type IN ('date', 'timestamp', 'timestamp without time zone', 'timestamptz');
    """
    )
    with engine.connect() as conn:
        date_column = conn.execute(query, {"table_name": table_name}).scalar()
        if not date_column:
            return None

        query = text(
            f"SELECT MIN({date_column}), MAX({date_column}) FROM {table_name};"
        )
        return conn.execute(query).fetchone()


@handle_exceptions
def get_table_row_counts(engine):
    """Retrieves row counts for each table."""
    query = text(
        """
        SELECT relname AS table_name, reltuples::bigint AS row_count
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE relkind = 'r' AND n.nspname = 'public'
        ORDER BY row_count DESC;
    """
    )
    with engine.connect() as conn:
        return conn.execute(query).fetchall()


def write_report(content):
    """Writes the database report to a markdown file."""
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write(content)


def generate_report():
    """Generates a detailed database analysis report."""
    engine = get_database_engine()
    if not engine:
        return

    report_content = f"# 📊 Database Report\n\n📅 **Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # ✅ Fetch Tables
    tables = get_tables(engine)
    report_content += "## 📌 List of Tables\n\n"
    report_content += "\n".join([f"- {table}" for table in tables]) + "\n\n"

    # ✅ Fetch Schema Details
    for table in tables:
        schema = get_table_schema(engine, table)
        report_content += f"## 📊 Schema of `{table}`\n\n"
        report_content += "| Column Name | Data Type | Nullable | Default Value |\n"
        report_content += "|------------|----------|----------|--------------|\n"
        report_content += (
            "\n".join(
                [f"| {col[0]} | {col[1]} | {col[2]} | {col[3]} |" for col in schema]
            )
            + "\n\n"
        )

    # ✅ Fetch Sample Data
    for table in tables:
        sample_data = get_sample_data(engine, table)
        report_content += f"## 🔍 Sample Data from `{table}`\n\n"
        if sample_data:
            columns = get_table_schema(engine, table)
            col_names = " | ".join([col[0] for col in columns])
            report_content += f"| {col_names} |\n"
            report_content += "| " + " | ".join(["---"] * len(columns)) + " |\n"
            report_content += (
                "\n".join(
                    [
                        "| " + " | ".join(str(x) if x else "NULL" for x in row) + " |"
                        for row in sample_data
                    ]
                )
                + "\n\n"
            )

    # ✅ Fetch Date Ranges
    report_content += "## 📅 Date Range of Stock Price Data\n\n"
    for table in tables:
        date_range = get_date_range(engine, table)
        if date_range and date_range[0] and date_range[1]:
            report_content += f"- **{table}**: Start Date = {date_range[0]}, End Date = {date_range[1]}\n"

    # ✅ Fetch Row Counts
    report_content += "## 📊 Row Counts for Each Table\n\n"
    row_counts = get_table_row_counts(engine)
    report_content += "| Table Name | Row Count |\n|------------|----------|\n"
    report_content += (
        "\n".join([f"| {row[0]} | {row[1]} |" for row in row_counts]) + "\n"
    )

    # ✅ Write Report
    write_report(report_content)
    console.print(f"\n✅ Report generated: [bold cyan]{REPORT_FILE}[/bold cyan]\n")


if __name__ == "__main__":
    generate_report()
