from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "kkbox" / "raw" / "train.csv"
TRANSACTIONS_FILE = PROJECT_ROOT / "data" / "kkbox" / "raw" / "transactions.csv"

con = duckdb.connect()


def fetch_scalar(query: str, default: object = 0):
    row = con.execute(query).fetchone()
    if row is None:
        return default
    return row[0]


print("\n=== TRAIN DATA ===")

train_count = fetch_scalar(f"""
    SELECT COUNT(*)
    FROM read_csv_auto('{TRAIN_FILE.as_posix()}')
""")

print(f"Rows: {train_count:,}")


train_unique = fetch_scalar(f"""
    SELECT COUNT(DISTINCT msno)
    FROM read_csv_auto('{TRAIN_FILE.as_posix()}')
""")

print(f"Unique customers: {train_unique:,}")


print("\n=== CHURN DISTRIBUTION ===")

churn_distribution = con.execute(f"""
    SELECT
        is_churn,
        COUNT(*) AS customers,
        ROUND(
            100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
            2
        ) AS percentage
    FROM read_csv_auto('{TRAIN_FILE.as_posix()}')
    GROUP BY is_churn
    ORDER BY is_churn
""").fetchdf()

print(churn_distribution.to_string(index=False))


# ---------------------------------------------------------
# TRANSACTIONS
# ---------------------------------------------------------
print("\n=== TRANSACTIONS ===")

transaction_count = fetch_scalar(f"""
    SELECT COUNT(*)
    FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
""")

print(f"Rows: {transaction_count:,}")


transaction_unique = fetch_scalar(f"""
    SELECT COUNT(DISTINCT msno)
    FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
""")

print(f"Unique customers: {transaction_unique:,}")


print("\n=== TRANSACTION DATE RANGE ===")

transaction_dates = con.execute(f"""
    SELECT
        MIN(transaction_date) AS min_transaction_date,
        MAX(transaction_date) AS max_transaction_date,
        MIN(membership_expire_date) AS min_expiry_date,
        MAX(membership_expire_date) AS max_expiry_date
    FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
""").fetchone()

if transaction_dates is None:
    transaction_dates = (None, None, None, None)

print(f"Minimum transaction date: {transaction_dates[0]}")
print(f"Maximum transaction date: {transaction_dates[1]}")
print(f"Minimum membership expiry: {transaction_dates[2]}")
print(f"Maximum membership expiry: {transaction_dates[3]}")

# LABELLED CUSTOMERS WITH / WITHOUT TRANSACTIONS
print("\n=== TRAIN / TRANSACTION COVERAGE ===")

coverage = con.execute(f"""
    WITH train AS (
        SELECT DISTINCT msno
        FROM read_csv_auto('{TRAIN_FILE.as_posix()}')
    ),
    transactions AS (
        SELECT DISTINCT msno
        FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
    )
    SELECT
        COUNT(*) AS labelled_customers,
        COUNT(t.msno) AS customers_with_transactions,
        COUNT(*) - COUNT(t.msno) AS customers_without_transactions,
        ROUND(
            100.0 * COUNT(t.msno) / COUNT(*),
            2
        ) AS pct_with_transactions
    FROM train tr
    LEFT JOIN transactions t
        ON tr.msno = t.msno
""").fetchone()

if coverage is None:
    coverage = (0, 0, 0, 0.0)

print(f"Labelled customers:              {coverage[0]:,}")
print(f"With transactions:               {coverage[1]:,}")
print(f"Without transactions:            {coverage[2]:,}")
print(f"% with transactions:             {coverage[3]:.2f}%")

# TRANSACTION SCHEMA
print("\n=== TRANSACTION SCHEMA ===")

schema = con.execute(f"""
    DESCRIBE
    SELECT *
    FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
""").fetchdf()

print(schema.to_string(index=False))

# SAMPLE TRANSACTIONS
print("\n=== TRANSACTION SAMPLE ===")

sample = con.execute(f"""
    SELECT *
    FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
    LIMIT 5
""").fetchdf()

print(sample.to_string(index=False))


con.close()