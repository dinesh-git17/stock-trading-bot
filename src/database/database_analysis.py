import os
from datetime import datetime

import psycopg2
from dotenv import load_dotenv
from rich.console import Console

# ✅ Load environment variables
load_dotenv()

# ✅ Setup console output
console = Console()

# ✅ Database Configuration (replace with actual credentials)
DB_NAME = os.getenv("DB_NAME", "your_database")
DB_USER = os.getenv("DB_USER", "your_username")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# ✅ Set output file
report_filename = "data/database_report.md"


# ✅ Connect to PostgreSQL with autocommit
def connect_db():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        conn.autocommit = True  # ✅ Prevent transaction block errors
        return conn
    except Exception as e:
        console.print(f"[bold red]Database connection failed![/bold red] {e}")
        return None


# ✅ Fetch all tables
def get_tables(conn):
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        return f"❌ Error retrieving table list: {e}"


# ✅ Fetch table schema
def get_table_schema(conn, table_name):
    query = f"""
    SELECT column_name, data_type, is_nullable, COALESCE(column_default::text, 'NULL') 
    FROM information_schema.columns 
    WHERE table_name = '{table_name}';
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
    except Exception as e:
        return f"❌ Error retrieving schema for {table_name}: {e}"


# ✅ Get first 5 rows of each table
def get_sample_data(conn, table_name):
    query = f"SELECT * FROM {table_name} LIMIT 5;"
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
    except Exception as e:
        return f"❌ Error retrieving sample data for {table_name}: {e}"


# ✅ Get stock price date range dynamically
def get_date_range(conn, table_name):
    try:
        # Find the correct date/timestamp column dynamically
        query = f"""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = '{table_name}' 
        AND data_type IN ('date', 'timestamp', 'timestamp without time zone', 'timestamptz');
        """
        with conn.cursor() as cur:
            cur.execute(query)
            date_column = cur.fetchone()

        if not date_column:
            return f"⚠️ No date column found in {table_name}"

        date_column = date_column[0]  # Extract column name

        # Query the date range
        query = f"SELECT MIN({date_column}) AS start_date, MAX({date_column}) AS end_date FROM {table_name};"
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchone()

    except Exception as e:
        return f"❌ Error retrieving date range for {table_name}: {e}"


# ✅ Get row count for each table
def get_table_row_counts(conn):
    query = """
    SELECT relname AS table_name, reltuples::bigint AS row_count
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE relkind = 'r' AND n.nspname = 'public'
    ORDER BY row_count DESC;
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
    except Exception as e:
        return f"❌ Error retrieving table row counts: {e}"


# ✅ Write report to file
def write_report(content):
    with open(report_filename, "w", encoding="utf-8") as file:
        file.write(content)


# ✅ Main function
def main():
    conn = connect_db()
    if not conn:
        return

    report_content = f"# 📊 Database Report\n\n📅 **Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # ✅ Get tables
    tables = get_tables(conn)
    report_content += f"## 📌 List of Tables\n\n"

    if isinstance(tables, str):  # Error handling
        report_content += f"{tables}\n\n"
    else:
        for table in tables:
            report_content += f"- {table}\n"

    # ✅ Show schema details
    for table in tables:
        report_content += f"\n## 📊 Schema of `{table}`\n\n"
        schema = get_table_schema(conn, table)

        if isinstance(schema, str):  # Error handling
            report_content += f"{schema}\n\n"
        else:
            report_content += "| Column Name | Data Type | Nullable | Default Value |\n"
            report_content += "|------------|----------|----------|--------------|\n"
            for col in schema:
                report_content += f"| {col[0]} | {col[1]} | {col[2]} | {col[3] if col[3] else 'NULL'} |\n"
            report_content += "\n"

    # ✅ Show sample data
    for table in tables:
        report_content += f"\n## 🔍 Sample Data from `{table}`\n\n"
        sample_data = get_sample_data(conn, table)

        if isinstance(sample_data, str):  # Error handling
            report_content += f"{sample_data}\n\n"
        else:
            report_content += (
                "| " + " | ".join(get_table_schema(conn, table)[0]) + " |\n"
            )
            report_content += "| " + " | ".join(["---"] * len(sample_data[0])) + " |\n"
            for row in sample_data:
                report_content += (
                    "| " + " | ".join(str(x) if x else "NULL" for x in row) + " |\n"
                )
            report_content += "\n"

    # ✅ Get stock price date range
    report_content += "\n## 📅 Date Range of Stock Price Data\n\n"
    for table in tables:
        date_range = get_date_range(conn, table)
        if isinstance(date_range, str):  # Error handling
            report_content += f"{date_range}\n"
        elif date_range[0] and date_range[1]:
            report_content += f"- **{table}**: Start Date = {date_range[0]}, End Date = {date_range[1]}\n"

    # ✅ Get row counts
    report_content += "\n## 📊 Row Counts for Each Table\n\n"
    row_counts = get_table_row_counts(conn)

    if isinstance(row_counts, str):  # Error handling
        report_content += f"{row_counts}\n"
    else:
        report_content += "| Table Name | Row Count |\n|------------|----------|\n"
        for row in row_counts:
            report_content += f"| {row[0]} | {row[1]} |\n"

    # ✅ Write to file
    write_report(report_content)
    console.print(f"\n✅ Report generated: [bold cyan]{report_filename}[/bold cyan]\n")

    # ✅ Close connection
    conn.close()


if __name__ == "__main__":
    main()
