from pathlib import Path
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "kkbox" / "raw" / "train.csv"
TRANSACTIONS_FILE = PROJECT_ROOT / "data" / "kkbox" / "raw" / "transactions.csv"


con = duckdb.connect()


print("\n=== SUSPICIOUS EXPIRY RECORDS ===")

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
        payment_method_id,
        payment_plan_days,
        plan_list_price,
        actual_amount_paid,
        is_auto_renew,
        transaction_date,
        membership_expire_date,
        is_cancel
    FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
)

SELECT
    COUNT(*) AS suspicious_transaction_records,
    COUNT(DISTINCT t.msno) AS suspicious_customers,
    COUNT(DISTINCT tr.msno) AS suspicious_labelled_customers
FROM transactions t
LEFT JOIN train tr
    ON t.msno = tr.msno
WHERE t.membership_expire_date < 20000101
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))


print("\n=== SUSPICIOUS RECORDS BY CANCELLATION STATUS ===")

query = f"""
WITH train AS (
    SELECT
        msno,
        is_churn
    FROM read_csv_auto('{TRAIN_FILE.as_posix()}')
),

transactions AS (
    SELECT *
    FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
)

SELECT
    t.is_cancel,
    COUNT(*) AS records,
    COUNT(DISTINCT t.msno) AS customers,
    COUNT(DISTINCT CASE
        WHEN tr.msno IS NOT NULL THEN t.msno
    END) AS labelled_customers
FROM transactions t
LEFT JOIN train tr
    ON t.msno = tr.msno
WHERE t.membership_expire_date < 20000101
GROUP BY t.is_cancel
ORDER BY t.is_cancel
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))


print("\n=== SUSPICIOUS RECORD SAMPLE ===")

query = f"""
SELECT *
FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
WHERE membership_expire_date < 20000101
LIMIT 20
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))


print("\n=== LABEL DISTRIBUTION FOR CUSTOMERS WITH SUSPICIOUS RECORDS ===")

query = f"""
WITH train AS (
    SELECT
        msno,
        is_churn
    FROM read_csv_auto('{TRAIN_FILE.as_posix()}')
),

suspicious_customers AS (
    SELECT DISTINCT msno
    FROM read_csv_auto('{TRANSACTIONS_FILE.as_posix()}')
    WHERE membership_expire_date < 20000101
)

SELECT
    tr.is_churn,
    COUNT(*) AS customers
FROM train tr
INNER JOIN suspicious_customers s
    ON tr.msno = s.msno
GROUP BY tr.is_churn
ORDER BY tr.is_churn
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))


con.close()