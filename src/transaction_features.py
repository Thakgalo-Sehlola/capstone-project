from pathlib import Path

import duckdb

# Project paths and observation window

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "kkbox" / "raw" / "train.csv"

TRANSACTIONS_FILE = (
    PROJECT_ROOT / "data" / "kkbox" / "raw" / "transactions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "kkbox" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "transaction_features.parquet"

WINDOW_START = "2016-12-01"
WINDOW_END = "2017-02-28"


def fetch_one_or_raise(result, description):
  row = result.fetchone()
  if row is None:
    raise RuntimeError(f"{description} query returned no rows.")
  return row


def main():
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  for file_path in [TRAIN_FILE, TRANSACTIONS_FILE]:
    if not file_path.exists():
      raise FileNotFoundError(f"Required file not found: {file_path}")

  con = duckdb.connect()

  # PERFORMANCE TUNING FOR 8GB RAM
  con.execute("PRAGMA memory_limit='3GB';")
  con.execute("PRAGMA threads=2;")

  try:

    # Load the labelled customer cohort

    con.execute(
        f"""
            CREATE OR REPLACE TEMP TABLE labelled_customers AS
            SELECT
                CAST(msno AS VARCHAR) AS msno,
                CAST(is_churn AS INTEGER) AS is_churn
            FROM read_csv_auto(
                '{TRAIN_FILE.as_posix()}',
                header = TRUE,
                all_varchar = TRUE
            )
            """
    )

    cohort_count = fetch_one_or_raise(
        con.execute(
            """
                SELECT
                    COUNT(*) AS row_count,
                    COUNT(DISTINCT msno) AS unique_customers
                FROM labelled_customers
                """
        ),
        "Labelled cohort",
    )

    if cohort_count[0] != cohort_count[1]:
      raise ValueError("Labelled cohort contains duplicate customer IDs.")

    print("\nLabelled cohort")
    print(f"Rows: {cohort_count[0]:,}")
    print(f"Unique customers: {cohort_count[1]:,}")


    # Read transaction data

    # Dates are supplied as YYYYMMDD values.
    # TRY_STRPTIME avoids interpreting invalid dates
    # as valid dates.

    con.execute(
        f"""
            CREATE OR REPLACE TEMP VIEW parsed_transactions AS
            SELECT
                CAST(msno AS VARCHAR) AS msno,

                TRY_CAST(payment_method_id AS INTEGER)
                    AS payment_method_id,

                TRY_CAST(payment_plan_days AS INTEGER)
                    AS payment_plan_days,

                TRY_CAST(plan_list_price AS DOUBLE)
                    AS plan_list_price,

                TRY_CAST(actual_amount_paid AS DOUBLE)
                    AS actual_amount_paid,

                TRY_CAST(is_auto_renew AS INTEGER)
                    AS is_auto_renew,

                TRY_STRPTIME(
                    CAST(transaction_date AS VARCHAR),
                    '%Y%m%d'
                )::DATE AS transaction_date,

                TRY_STRPTIME(
                    CAST(membership_expire_date AS VARCHAR),
                    '%Y%m%d'
                )::DATE AS membership_expire_date,

                TRY_CAST(is_cancel AS INTEGER)
                    AS is_cancel

            FROM read_csv_auto(
                '{TRANSACTIONS_FILE.as_posix()}',
                header = TRUE,
                all_varchar = TRUE
            )
            """
    )


    # Profile date and categorical values

    source_checks = con.execute(
        """
            SELECT
                COUNT(*) AS total_rows,

                COUNT(*) FILTER (
                    WHERE transaction_date IS NULL
                ) AS invalid_transaction_dates,

                COUNT(*) FILTER (
                    WHERE membership_expire_date IS NULL
                ) AS invalid_expiry_dates,

                COUNT(*) FILTER (
                    WHERE transaction_date BETWEEN
                        DATE '2016-12-01'
                        AND DATE '2017-02-28'
                ) AS rows_in_window

            FROM parsed_transactions
            """
    ).fetchone() or (0, 0, 0, 0)

    print("\nTransaction source checks")
    print(f"Source rows: {source_checks[0]:,}")
    print(f"Invalid transaction dates: {source_checks[1]:,}")
    print(f"Invalid expiry dates: {source_checks[2]:,}")
    print(f"Rows in window: {source_checks[3]:,}")

    print("\nAuto-renew values in window")
    auto_values = con.execute(
        """
            SELECT
                is_auto_renew,
                COUNT(*) AS records
            FROM parsed_transactions
            WHERE transaction_date BETWEEN
                DATE '2016-12-01'
                AND DATE '2017-02-28'
            GROUP BY is_auto_renew
            ORDER BY is_auto_renew
            """
    ).fetchall()

    for value, count in auto_values:
      print(f"{value}: {count:,}")

    print("\nCancellation values in window")
    cancel_values = con.execute(
        """
            SELECT
                is_cancel,
                COUNT(*) AS records
            FROM parsed_transactions
            WHERE transaction_date BETWEEN
                DATE '2016-12-01'
                AND DATE '2017-02-28'
            GROUP BY is_cancel
            ORDER BY is_cancel
            """
    ).fetchall()

    for value, count in cancel_values:
      print(f"{value}: {count:,}")

    # Filter the observation window and aggregate
    con.execute(
        """
            CREATE OR REPLACE TEMP TABLE transaction_agg AS
            SELECT
                msno,

                COUNT(*) AS transaction_count,

                SUM(actual_amount_paid)
                    AS total_amount_paid,

                AVG(actual_amount_paid)
                    AS avg_amount_paid,

                SUM(plan_list_price)
                    AS total_plan_list_price,

                AVG(payment_plan_days)
                    AS avg_payment_plan_days,

                SUM(
                    CASE
                        WHEN is_auto_renew = 1 THEN 1
                        ELSE 0
                    END
                )::DOUBLE / NULLIF(COUNT(*), 0)
                    AS auto_renew_rate,

                SUM(
                    CASE
                        WHEN is_cancel = 1 THEN 1
                        ELSE 0
                    END
                )::DOUBLE / NULLIF(COUNT(*), 0)
                    AS cancellation_rate,

                DATE_DIFF(
                    'day',
                    MAX(transaction_date),
                    DATE '2017-02-28'
                ) AS transaction_recency_days,

                COUNT(*) FILTER (
                    WHERE membership_expire_date
                        >= DATE '2000-01-01'
                ) AS valid_expiry_count

            FROM parsed_transactions

            WHERE transaction_date BETWEEN
                DATE '2016-12-01'
                AND DATE '2017-02-28'

            GROUP BY msno
            """
    )

    # Join to the labelled cohort

    con.execute(
        """
            CREATE OR REPLACE TEMP TABLE transaction_features AS
            SELECT
                c.msno,
                c.is_churn,

                COALESCE(
                    t.transaction_count, 0
                ) AS transaction_count,

                COALESCE(
                    t.total_amount_paid, 0.0
                ) AS total_amount_paid,

                t.avg_amount_paid,

                COALESCE(
                    t.total_plan_list_price, 0.0
                ) AS total_plan_list_price,

                t.avg_payment_plan_days,

                t.auto_renew_rate,
                t.cancellation_rate,
                t.transaction_recency_days,

                COALESCE(
                    t.valid_expiry_count, 0
                ) AS valid_expiry_count

            FROM labelled_customers c

            LEFT JOIN transaction_agg t
                ON c.msno = t.msno
            """
    )

    # Validate the output

    checks = fetch_one_or_raise(
        con.execute(
            """
                SELECT
                    COUNT(*) AS row_count,
                    COUNT(DISTINCT msno) AS unique_customers,

                    COUNT(*) FILTER (
                        WHERE msno IS NULL
                    ) AS null_ids,

                    COUNT(*) FILTER (
                        WHERE is_churn NOT IN (0, 1)
                           OR is_churn IS NULL
                    ) AS invalid_targets,

                    COUNT(*) FILTER (
                        WHERE transaction_count < 0
                           OR valid_expiry_count < 0
                    ) AS negative_counts,

                    COUNT(*) FILTER (
                        WHERE auto_renew_rate < 0
                           OR auto_renew_rate > 1
                           OR cancellation_rate < 0
                           OR cancellation_rate > 1
                    ) AS invalid_rates,

                    COUNT(*) FILTER (
                        WHERE transaction_recency_days < 0
                    ) AS negative_recency,

                    COUNT(*) FILTER (
                        WHERE transaction_count = 0
                          AND (
                              avg_amount_paid IS NOT NULL
                              OR avg_payment_plan_days IS NOT NULL
                              OR auto_renew_rate IS NOT NULL
                              OR cancellation_rate IS NOT NULL
                              OR transaction_recency_days IS NOT NULL
                          )
                    ) AS inconsistent_empty_windows

                FROM transaction_features
                """
        ),
        "Transaction feature validation",
    )

    print("\nTransaction feature validation")
    print(f"Rows: {checks[0]:,}")
    print(f"Unique customers: {checks[1]:,}")
    print(f"Null IDs: {checks[2]:,}")
    print(f"Invalid targets: {checks[3]:,}")
    print(f"Negative counts: {checks[4]:,}")
    print(f"Invalid rates: {checks[5]:,}")
    print(f"Negative recency: {checks[6]:,}")
    print(f"Inconsistent empty windows: {checks[7]:,}")

    if checks[0] != cohort_count[0]:
      raise ValueError("Output row count differs from labelled cohort.")

    if checks[1] != cohort_count[1]:
      raise ValueError("Output does not contain one row per customer.")

    if any(checks[i] > 0 for i in range(2, 8)):
      raise ValueError("Transaction feature validation failed.")

    # Save the output

    con.execute(
        f"""
            COPY transaction_features
            TO '{OUTPUT_FILE.as_posix()}'
            (FORMAT PARQUET, COMPRESSION SNAPPY)
            """
    )

    print(f"\nSaved: {OUTPUT_FILE}")
    print("Transaction feature engineering completed.")

  finally:
    con.close()


if __name__ == "__main__":
  main()