from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "kkbox" / "processed"

BEHAVIOUR_FILE = PROCESSED_DIR / "behaviour_features.parquet"
TRANSACTION_FILE = PROCESSED_DIR / "transaction_features.parquet"
MEMBER_FILE = PROCESSED_DIR / "member_features.parquet"

OUTPUT_FILE = PROCESSED_DIR / "kkbox_model_features.parquet"


def main():
  required_files = [
      BEHAVIOUR_FILE,
      TRANSACTION_FILE,
      MEMBER_FILE,
  ]

  for file_path in required_files:
    if not file_path.exists():
      raise FileNotFoundError(f"Required feature file not found: {file_path}")

  con = duckdb.connect()

  # Performance tuning for 8GB RAM
  con.execute("PRAGMA memory_limit='3GB';")
  con.execute("PRAGMA threads=2;")

  try:
    # Read the feature tables
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW behaviour AS
        SELECT *
        FROM read_parquet('{BEHAVIOUR_FILE.as_posix()}')
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW transactions AS
        SELECT *
        FROM read_parquet('{TRANSACTION_FILE.as_posix()}')
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW members AS
        SELECT *
        FROM read_parquet('{MEMBER_FILE.as_posix()}')
        """
    )

    # Validate each source table
    source_names = [
        "behaviour",
        "transactions",
        "members",
    ]

    source_counts = {}

    for source in source_names:
      result = con.execute(
          f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT msno) AS unique_customers,
                COUNT(*) FILTER (
                    WHERE msno IS NULL
                ) AS null_ids
            FROM {source}
            """
      ).fetchone()

      if result is None:
        raise ValueError(f"{source} query returned no rows. Check the source table.")

      source_counts[source] = result[0]

      print(f"\n{source} source checks")
      print(f"Rows: {result[0]:,}")
      print(f"Unique customers: {result[1]:,}")
      print(f"Null IDs: {result[2]:,}")

      if result[0] != result[1]:
        raise ValueError(f"{source} contains duplicate customer IDs.")

      if result[2] > 0:
        raise ValueError(f"{source} contains null customer IDs.")

    # Each feature table was built from the same labelled cohort.
    # Confirm the row counts agree.
    if len(set(source_counts.values())) != 1:
      raise ValueError("Feature tables have different row counts.")

    # Check target agreement across sources
    target_checks = con.execute(
        """
        SELECT
            COUNT(*) FILTER (
                WHERE b.is_churn != t.is_churn
                   OR b.is_churn != m.is_churn
                   OR b.is_churn IS NULL
                   OR t.is_churn IS NULL
                   OR m.is_churn IS NULL
            ) AS target_disagreements

        FROM behaviour b

        INNER JOIN transactions t
            ON b.msno = t.msno

        INNER JOIN members m
            ON b.msno = m.msno
        """
    ).fetchone()

    if target_checks is None:
      raise ValueError("Target agreement query returned no rows.")

    print("\nTarget agreement")
    print("Target disagreements: " f"{target_checks[0]:,}")

    if target_checks[0] != 0:
      raise ValueError("The target differs between feature tables.")

    # Join feature groups
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE model_features AS
        SELECT
            b.msno,
            b.is_churn,

            -- Behavioural predictors
            b.active_days_M1,
            b.total_secs_M1,
            b.total_num_unq_M1,
            b.total_song_events_M1,
            b.avg_daily_secs_M1,

            b.active_days_M2,
            b.total_secs_M2,
            b.total_num_unq_M2,
            b.total_song_events_M2,
            b.avg_daily_secs_M2,

            b.active_days_M3,
            b.total_secs_M3,
            b.total_num_unq_M3,
            b.total_song_events_M3,
            b.avg_daily_secs_M3,

            -- Transaction predictors
            t.transaction_count,
            t.total_amount_paid,
            t.avg_amount_paid,
            t.total_plan_list_price,
            t.avg_payment_plan_days,
            t.auto_renew_rate,
            t.cancellation_rate,
            t.transaction_recency_days,
            t.valid_expiry_count,

            -- Member predictors
            m.city,
            m.gender,
            m.age,
            m.registered_via,
            m.registration_year,
            m.registration_month

        FROM behaviour b

        INNER JOIN transactions t
            ON b.msno = t.msno

        INNER JOIN members m
            ON b.msno = m.msno
        """
    )

    # Validate combined dataset
    checks = con.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT msno) AS unique_customers,

            COUNT(*) FILTER (
                WHERE msno IS NULL
            ) AS null_ids,

            COUNT(*) FILTER (
                WHERE is_churn IS NULL
                   OR is_churn NOT IN (0, 1)
            ) AS invalid_targets,

            COUNT(*) FILTER (
                WHERE transaction_count < 0
                   OR valid_expiry_count < 0
                   OR active_days_M1 < 0
                   OR active_days_M2 < 0
                   OR active_days_M3 < 0
            ) AS invalid_counts

        FROM model_features
        """
    ).fetchone()

    if checks is None:
      raise ValueError("Combined dataset query returned no rows. Check the joined model_features table.")

    print("\nCombined dataset validation")
    print(f"Rows: {checks[0]:,}")
    print(f"Unique customers: {checks[1]:,}")
    print(f"Null IDs: {checks[2]:,}")
    print(f"Invalid targets: {checks[3]:,}")
    print(f"Invalid counts: {checks[4]:,}")

    if checks[0] != source_counts["behaviour"]:
      raise ValueError("Combined row count differs from source cohort.")

    if checks[1] != checks[0]:
      raise ValueError("Combined dataset has duplicate customer IDs.")

    if any(checks[i] > 0 for i in range(2, 5)):
      raise ValueError("Combined dataset validation failed.")

    # Target distribution
    print("\nTarget distribution")

    distribution = con.execute(
        """
        SELECT
            is_churn,
            COUNT(*) AS customers,
            ROUND(
                100.0 * COUNT(*) /
                SUM(COUNT(*)) OVER (),
                2
            ) AS percentage
        FROM model_features
        GROUP BY is_churn
        ORDER BY is_churn
        """
    ).fetchall()

    for target, count, percentage in distribution:
      print(f"is_churn={target}: {count:,} customers ({percentage}%)")

    # Feature count and schema
    schema = con.execute(
        """
        DESCRIBE SELECT *
        FROM model_features
        """
    ).fetchall()

    print(f"\nTotal columns: {len(schema)}")
    print("Columns:")
    for column in schema:
      print(f"  {column[0]}: {column[1]}")

    # Write combined modelling dataset
    con.execute(
        f"""
        COPY model_features
        TO '{OUTPUT_FILE.as_posix()}'
        (FORMAT PARQUET, COMPRESSION SNAPPY)
        """
    )

    print(f"\nSaved: {OUTPUT_FILE}")
    print("Combined modelling dataset created.")

  finally:
    con.close()


if __name__ == "__main__":
  main()