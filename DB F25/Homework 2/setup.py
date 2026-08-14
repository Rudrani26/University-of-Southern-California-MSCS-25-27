# setup_db.py
import duckdb
import os

from tabulate import tabulate


# --- 1️⃣ Connect to DuckDB (in-memory or on-disk) ---
# For persistence, replace ':memory:' with 'summer_camp.db'
con = duckdb.connect(database=':memory:')

# --- 2️⃣ Drop all existing tables (clean slate) ---
existing_tables = con.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'main';
""").fetchall()

for (table_name,) in existing_tables:
    con.execute(f"DROP TABLE IF EXISTS {table_name};")

# --- 3️⃣ Define CSV load function ---
def load_csv(table_name, filename):
    path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(path):
        print(f"⚠️  File not found: {filename} (skipped)")
        return
    con.execute(f"""
        CREATE TABLE {table_name} AS
        SELECT * FROM read_csv_auto(
            '{path}',
            header = TRUE,
            delim = ',',
            all_varchar = TRUE,
            ignore_errors = TRUE
        );
    """)

# --- 4️⃣ Load all CSVs ---
csv_files = {
    'PERSON': 'PERSON.csv',
    'COURSE': 'COURSE.csv',
    'CODINGCLASS': 'CODINGCLASS.csv',
    'GROUP_ENROLMENT': 'GROUP_ENROLMENT.csv',
    'RATING_INSTRUCTOR': 'RATING_INSTRUCTOR.csv',
    'PROJECT_SUPERVISOR': 'PROJECT_SUPERVISOR.csv',
    'PROJECT_GROUP': 'PROJECT_GROUP.csv'
}

for table, file in csv_files.items():
    load_csv(table, file)

def print_table(df):
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
