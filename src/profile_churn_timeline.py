from pathlib import Path
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "kkbox" / "raw" / "train.csv"
TRANSACTIONS_FILE = PROJECT_ROOT / "data" / "kkbox" / "raw" / "transactions.csv"


con = duckdb.connect()


print("\n=== LABELLED CUSTOMER TIMELINE ===")

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
        transaction_date,
        membership_expire_date,
        actual_amount_paid,
        is_auto_renew,
        is_cancel
    FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
),

customer_timeline AS (
    SELECT
        t.msno,
        t.is_churn,
        MIN(x.transaction_date) AS first_transaction_date,
        MAX(x.transaction_date) AS last_transaction_date,
        MIN(x.membership_expire_date) AS first_expiry_date,
        MAX(x.membership_expire_date) AS last_expiry_date,
        COUNT(*) AS transaction_count
    FROM train t
    INNER JOIN transactions x
        ON t.msno = x.msno
    GROUP BY
        t.msno,
        t.is_churn
)

SELECT *
FROM customer_timeline
LIMIT 10
"""

sample = con.execute(query).fetchdf()

print(sample.to_string(index=False))


print("\n=== TIMELINE SUMMARY ===")

summary_query = f"""
WITH train AS (
    SELECT
        msno,
        is_churn
    FROM read_csv_auto('{TRAIN_FILE.as_posix()}')
),

transactions AS (
    SELECT
        msno,
        transaction_date,
        membership_expire_date
    FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
),

customer_timeline AS (
    SELECT
        t.msno,
        t.is_churn,
        MIN(x.transaction_date) AS first_transaction_date,
        MAX(x.transaction_date) AS last_transaction_date,
        MAX(x.membership_expire_date) AS last_expiry_date
    FROM train t
    INNER JOIN transactions x
        ON t.msno = x.msno
    GROUP BY
        t.msno,
        t.is_churn
)

SELECT
    is_churn,
    COUNT(*) AS customers,

    MIN(last_transaction_date) AS min_last_transaction,
    MAX(last_transaction_date) AS max_last_transaction,

    MIN(last_expiry_date) AS min_last_expiry,
    MAX(last_expiry_date) AS max_last_expiry

FROM customer_timeline

GROUP BY is_churn

ORDER BY is_churn
"""

summary = con.execute(summary_query).fetchdf()

print(summary.to_string(index=False))


print("\n=== EXPIRY DATE QUALITY CHECK ===")

expiry_query = f"""
SELECT
    COUNT(*) AS total_transactions,

    SUM(
        CASE
            WHEN membership_expire_date < 20000101
            THEN 1
            ELSE 0
        END
    ) AS suspicious_expiry_dates,

    MIN(membership_expire_date) AS minimum_expiry,
    MAX(membership_expire_date) AS maximum_expiry

FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
"""

expiry = con.execute(expiry_query).fetchdf()

print(expiry.to_string(index=False))


con.close()