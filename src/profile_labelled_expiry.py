from pathlib import Path
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "kkbox" / "raw" / "train.csv"
TRANSACTIONS_FILE = PROJECT_ROOT / "data" / "kkbox" / "raw" / "transactions.csv"


con = duckdb.connect()


print("\n=== LABELLED CUSTOMERS: LAST VALID EXPIRY ===")

query = f"""
WITH train AS (
    SELECT
        msno,
        is_churn
    FROM read_csv_auto('{TRAIN_FILE.as_posix()}')
),

transactions AS (
    SELECT
        msno,
        membership_expire_date
    FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
    WHERE membership_expire_date >= 20000101
),

last_expiry AS (
    SELECT
        msno,
        MAX(membership_expire_date) AS last_valid_expiry
    FROM transactions
    GROUP BY msno
)

SELECT
    tr.is_churn,
    COUNT(*) AS customers,
    MIN(le.last_valid_expiry) AS minimum_last_valid_expiry,
    MAX(le.last_valid_expiry) AS maximum_last_valid_expiry
FROM train tr
INNER JOIN last_expiry le
    ON tr.msno = le.msno
GROUP BY tr.is_churn
ORDER BY tr.is_churn
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))


print("\n=== LAST VALID EXPIRY DISTRIBUTION ===")

query = f"""
WITH train AS (
    SELECT
        msno,
        is_churn
    FROM read_csv_auto('{TRAIN_FILE.as_posix()}')
),

transactions AS (
    SELECT
        msno,
        membership_expire_date
    FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
    WHERE membership_expire_date >= 20000101
),

last_expiry AS (
    SELECT
        msno,
        MAX(membership_expire_date) AS last_valid_expiry
    FROM transactions
    GROUP BY msno
)

SELECT
    last_valid_expiry,
    COUNT(*) AS customers
FROM train tr
INNER JOIN last_expiry le
    ON tr.msno = le.msno
GROUP BY last_valid_expiry
ORDER BY last_valid_expiry
"""

result = con.execute(query).fetchdf()

print(result.to_string(index=False))


print("\n=== LABELLED CUSTOMERS WITH FEBRUARY 2017 EXPIRY ===")

query = f"""
WITH train AS (
    SELECT
        msno,
        is_churn
    FROM read_csv_auto('{TRAIN_FILE.as_posix()}')
),

transactions AS (
    SELECT
        msno,
        membership_expire_date
    FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
    WHERE membership_expire_date >= 20170201
      AND membership_expire_date <= 20170228
),

february_customers AS (
    SELECT DISTINCT msno
    FROM transactions
)

SELECT
    tr.is_churn,
    COUNT(*) AS customers
FROM train tr
INNER JOIN february_customers fc
    ON tr.msno = fc.msno
GROUP BY tr.is_churn
ORDER BY tr.is_churn
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))


con.close()