from pathlib import Path

import duckdb

# Project paths and observation window
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "kkbox" / "raw" / "train.csv"

LOG_PARQUET_DIR = (
    PROJECT_ROOT / "data" / "kkbox" / "processed" / "log_chunks"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "kkbox" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "behaviour_features.parquet"

# Feature observation window
WINDOW_START = "2016-12-01"
WINDOW_END = "2017-02-28"

# Read the Parquet files directly using DuckDB
PARQUET_GLOB = (LOG_PARQUET_DIR / "user_logs_*.parquet").as_posix()

def main():
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

  if not LOG_PARQUET_DIR.exists():
    raise FileNotFoundError(f"Log directory not found: {LOG_PARQUET_DIR}")

  con = duckdb.connect()

  # Limit DuckDB to 3GB so it leaves room for the operating sytem and other apps
  con.execute("PRAGMA memory_limit='3GB';")
    
  # Restrict threads so it doesn't max out the CPU
  con.execute("PRAGMA threads=2;")

  try:
    # Load labelled customer IDs and target
    train_path = TRAIN_FILE.as_posix()

    con.execute(
        """
            CREATE OR REPLACE TEMP TABLE labelled_customers AS
            SELECT
                CAST(msno AS VARCHAR) AS msno,
                CAST(is_churn AS INTEGER) AS is_churn
            FROM read_csv_auto(
                ?,
                header = TRUE,
                all_varchar = TRUE
            )
            """,
        [train_path],
    )

    # Check that the labelled data has one row per user
    train_checks = con.execute(
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
                ) AS invalid_targets
            FROM labelled_customers
            """
    ).fetchone()

    if train_checks is None:
      raise ValueError("Labelled customer validation query returned no rows.")

    print("\nLabelled customer checks")
    print(f"Rows:              {train_checks[0]:,}")
    print(f"Unique customers:  {train_checks[1]:,}")
    print(f"Null IDs:          {train_checks[2]:,}")
    print(f"Invalid targets:   {train_checks[3]:,}")

    if train_checks[0] != train_checks[1]:
      raise ValueError(
          "train.csv does not contain exactly one row per customer."
      )

    if train_checks[2] > 0 or train_checks[3] > 0:
      raise ValueError(
          "Null customer IDs or invalid target values were found in train.csv."
      )

    # Read and filter the log data
    con.execute(
        f"""
            CREATE OR REPLACE TEMP VIEW parsed_logs AS
            SELECT
                CAST(msno AS VARCHAR) AS msno,
                TRY_STRPTIME(
                    CAST(date AS VARCHAR),
                    '%Y%m%d'
                )::DATE AS log_date,
                CAST(num_25 AS BIGINT) AS num_25,
                CAST(num_50 AS BIGINT) AS num_50,
                CAST(num_75 AS BIGINT) AS num_75,
                CAST(num_985 AS BIGINT) AS num_985,
                CAST(num_100 AS BIGINT) AS num_100,
                CAST(num_unq AS BIGINT) AS num_unq,
                CAST(total_secs AS DOUBLE) AS total_secs
            FROM read_parquet('{PARQUET_GLOB}')
            """
    )

    date_checks = con.execute(
        """
            SELECT
                COUNT(*) AS total_rows,
                COUNT(*) FILTER (
                    WHERE log_date IS NULL
                ) AS invalid_dates,
                COUNT(*) FILTER (
                    WHERE log_date BETWEEN
                        DATE '2016-12-01'
                        AND DATE '2017-02-28'
                ) AS rows_in_window
            FROM parsed_logs
            """
    ).fetchone()

    if date_checks is None:
      raise RuntimeError("No date-check row was returned from parsed_logs.")

    print("\nLog date checks")
    print(f"Total log rows:    {date_checks[0]:,}")
    print(f"Invalid dates:     {date_checks[1]:,}")
    print(f"Rows in window:    {date_checks[2]:,}")

    # Aggregate logs to one row per customer
    con.execute(
        """
            CREATE OR REPLACE TEMP TABLE monthly_behaviour AS
            WITH window_logs AS (
                SELECT *
                FROM parsed_logs
                WHERE log_date BETWEEN
                    DATE '2016-12-01'
                    AND DATE '2017-02-28'
            )

            SELECT
                msno,

                -- M1: February 2017
                COUNT(DISTINCT log_date) FILTER (
                    WHERE log_date >= DATE '2017-02-01'
                      AND log_date <  DATE '2017-03-01'
                ) AS active_days_M1,

                SUM(total_secs) FILTER (
                    WHERE log_date >= DATE '2017-02-01'
                      AND log_date <  DATE '2017-03-01'
                ) AS total_secs_M1,

                SUM(num_unq) FILTER (
                    WHERE log_date >= DATE '2017-02-01'
                      AND log_date <  DATE '2017-03-01'
                ) AS total_num_unq_M1,

                SUM(
                    num_25 + num_50 + num_75
                    + num_985 + num_100
                ) FILTER (
                    WHERE log_date >= DATE '2017-02-01'
                      AND log_date <  DATE '2017-03-01'
                ) AS total_song_events_M1,

                -- M2: January 2017
                COUNT(DISTINCT log_date) FILTER (
                    WHERE log_date >= DATE '2017-01-01'
                      AND log_date <  DATE '2017-02-01'
                ) AS active_days_M2,

                SUM(total_secs) FILTER (
                    WHERE log_date >= DATE '2017-01-01'
                      AND log_date <  DATE '2017-02-01'
                ) AS total_secs_M2,

                SUM(num_unq) FILTER (
                    WHERE log_date >= DATE '2017-01-01'
                      AND log_date <  DATE '2017-02-01'
                ) AS total_num_unq_M2,

                SUM(
                    num_25 + num_50 + num_75
                    + num_985 + num_100
                ) FILTER (
                    WHERE log_date >= DATE '2017-01-01'
                      AND log_date <  DATE '2017-02-01'
                ) AS total_song_events_M2,

                -- M3: December 2016
                COUNT(DISTINCT log_date) FILTER (
                    WHERE log_date >= DATE '2016-12-01'
                      AND log_date <  DATE '2017-01-01'
                ) AS active_days_M3,

                SUM(total_secs) FILTER (
                    WHERE log_date >= DATE '2016-12-01'
                      AND log_date <  DATE '2017-01-01'
                ) AS total_secs_M3,

                SUM(num_unq) FILTER (
                    WHERE log_date >= DATE '2016-12-01'
                      AND log_date <  DATE '2017-01-01'
                ) AS total_num_unq_M3,

                SUM(
                    num_25 + num_50 + num_75
                    + num_985 + num_100
                ) FILTER (
                    WHERE log_date >= DATE '2016-12-01'
                      AND log_date <  DATE '2017-01-01'
                ) AS total_song_events_M3

            FROM window_logs
            GROUP BY msno
            """
    )

    # Join to all labelled customers
    con.execute(
        """
            CREATE OR REPLACE TEMP TABLE behaviour_features AS
            SELECT
                c.msno,
                c.is_churn,

                COALESCE(b.active_days_M1, 0)
                    AS active_days_M1,
                COALESCE(b.total_secs_M1, 0.0)
                    AS total_secs_M1,
                COALESCE(b.total_num_unq_M1, 0)
                    AS total_num_unq_M1,
                COALESCE(b.total_song_events_M1, 0)
                    AS total_song_events_M1,

                COALESCE(b.active_days_M2, 0)
                    AS active_days_M2,
                COALESCE(b.total_secs_M2, 0.0)
                    AS total_secs_M2,
                COALESCE(b.total_num_unq_M2, 0)
                    AS total_num_unq_M2,
                COALESCE(b.total_song_events_M2, 0)
                    AS total_song_events_M2,

                COALESCE(b.active_days_M3, 0)
                    AS active_days_M3,
                COALESCE(b.total_secs_M3, 0.0)
                    AS total_secs_M3,
                COALESCE(b.total_num_unq_M3, 0)
                    AS total_num_unq_M3,
                COALESCE(b.total_song_events_M3, 0)
                    AS total_song_events_M3,

                b.total_secs_M1
                    / NULLIF(b.active_days_M1, 0)
                    AS avg_daily_secs_M1,

                b.total_secs_M2
                    / NULLIF(b.active_days_M2, 0)
                    AS avg_daily_secs_M2,

                b.total_secs_M3
                    / NULLIF(b.active_days_M3, 0)
                    AS avg_daily_secs_M3

            FROM labelled_customers c
            LEFT JOIN monthly_behaviour b
                ON c.msno = b.msno
            """
    )

    # Validate the output before writing
    output_checks = con.execute(
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
                    WHERE active_days_M1 < 0
                       OR active_days_M2 < 0
                       OR active_days_M3 < 0
                ) AS invalid_active_days,
                COUNT(*) FILTER (
                    WHERE avg_daily_secs_M1 < 0
                       OR avg_daily_secs_M2 < 0
                       OR avg_daily_secs_M3 < 0
                ) AS negative_daily_averages
            FROM behaviour_features
            """
    ).fetchone()

    if output_checks is None:
      raise ValueError("Feature output validation query returned no rows.")

    print("\nFeature output checks")
    print(f"Rows:                 {output_checks[0]:,}")
    print(f"Unique customers:     {output_checks[1]:,}")
    print(f"Null IDs:             {output_checks[2]:,}")
    print(f"Invalid targets:      {output_checks[3]:,}")
    print(f"Invalid active days:  {output_checks[4]:,}")
    print(f"Negative averages:    {output_checks[5]:,}")

    if output_checks[0] != train_checks[0]:
      raise ValueError("Output row count does not match train.csv.")

    if output_checks[1] != train_checks[1]:
      raise ValueError("Output does not have one row per labelled customer.")

    if any(output_checks[i] > 0 for i in range(2, 6)):
      raise ValueError("Feature validation failed. Review the checks.")

    # Write the customer-level feature dataset
    con.execute(
        f"""
            COPY behaviour_features
            TO '{OUTPUT_FILE.as_posix()}'
            (FORMAT PARQUET, COMPRESSION SNAPPY)
            """
    )

    print(f"\nSaved: {OUTPUT_FILE}")
    print("Behavioural feature engineering completed.")

  finally:
    con.close()


if __name__ == "__main__":
  main()