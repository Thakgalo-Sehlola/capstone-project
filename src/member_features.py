from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "kkbox" / "raw" / "train.csv"

MEMBERS_FILE = PROJECT_ROOT / "data" / "kkbox" / "raw" / "members_v3.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "kkbox" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "member_features.parquet"


def main():
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  for file_path in [TRAIN_FILE, MEMBERS_FILE]:
    if not file_path.exists():
      raise FileNotFoundError(f"Required file not found: {file_path}")

  con = duckdb.connect()

  # Performance tuning for 8GB RAM
  con.execute("PRAGMA memory_limit='3GB';")
  con.execute("PRAGMA threads=2;")

  try:
    # Load labelled cohort
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

    cohort_count = con.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT msno) AS unique_customers
        FROM labelled_customers
        """
    ).fetchone()

    if cohort_count is None:
        raise ValueError("Labelled cohort is empty or could not be loaded.")

    if cohort_count[0] != cohort_count[1]:
        raise ValueError("Labelled cohort contains duplicate IDs.")

    # Read member records
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW raw_members AS
        SELECT
            CAST(msno AS VARCHAR) AS msno,
            TRY_CAST(city AS INTEGER) AS city,
            NULLIF(TRIM(gender), '') AS gender,
            TRY_CAST(bd AS INTEGER) AS bd,
            TRY_CAST(registered_via AS INTEGER)
                AS registered_via,
            TRY_CAST(
                registration_init_time AS BIGINT
            ) AS registration_init_time
        FROM read_csv_auto(
            '{MEMBERS_FILE.as_posix()}',
            header = TRUE,
            all_varchar = TRUE
        )
        """
    )

    # Check source uniqueness and profile values
    member_checks = con.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT msno) AS unique_customers,
            COUNT(*) FILTER (
                WHERE msno IS NULL
            ) AS null_ids
        FROM raw_members
        """
    ).fetchone()

    if member_checks is None:
      raise ValueError("Member source check query returned no results.")

    print("\nMember source checks")
    print(f"Rows: {member_checks[0]:,}")
    print(f"Unique customers: {member_checks[1]:,}")
    print(f"Null IDs: {member_checks[2]:,}")

    if member_checks[0] != member_checks[1]:
      raise ValueError(
          "members_v3.csv has duplicate customer IDs. "
          "Investigate before joining."
      )

    if member_checks[2] > 0:
      raise ValueError("Null customer IDs found in member data.")

    print("\nGender values")
    for value, count in con.execute(
        """
        SELECT gender, COUNT(*)
        FROM raw_members
        GROUP BY gender
        ORDER BY gender
        """
    ).fetchall():
      print(f"{value}: {count:,}")

    print("\nCity code counts (top 20)")
    for value, count in con.execute(
        """
        SELECT city, COUNT(*)
        FROM raw_members
        GROUP BY city
        ORDER BY COUNT(*) DESC
        LIMIT 20
        """
    ).fetchall():
      print(f"{value}: {count:,}")

    print("\nRegistration channel counts")
    for value, count in con.execute(
        """
        SELECT registered_via, COUNT(*)
        FROM raw_members
        GROUP BY registered_via
        ORDER BY registered_via
        """
    ).fetchall():
      print(f"{value}: {count:,}")

    # Parse registration date and clean age
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW cleaned_members AS
        WITH parsed AS (
            SELECT
                msno,
                city,
                gender,
                bd,
                registered_via,

                TRY_STRPTIME(
                    CAST(registration_init_time AS VARCHAR),
                    '%Y%m%d'
                )::DATE AS registration_date

            FROM raw_members
        )

        SELECT
            msno,

            city,

            gender,

            CASE
                WHEN bd BETWEEN 5 AND 99
                THEN bd
                ELSE NULL
            END AS age,

            registered_via,

            YEAR(registration_date)
                AS registration_year,

            MONTH(registration_date)
                AS registration_month

        FROM parsed
        """
    )

    date_checks = con.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(*) FILTER (
                WHERE registration_year IS NULL
            ) AS invalid_registration_dates,
            COUNT(*) FILTER (
                WHERE age IS NULL
            ) AS missing_or_invalid_age
        FROM cleaned_members
        """
    ).fetchone()

    if date_checks is None:
        raise ValueError("Member cleaning check query returned no results.")

    print("\nMember cleaning checks")
    print(f"Rows: {date_checks[0]:,}")
    print("Invalid registration dates: " f"{date_checks[1]:,}")
    print("Missing/invalid age: " f"{date_checks[2]:,}")

    # Join to labelled customers
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE member_features AS
        SELECT
            c.msno,
            c.is_churn,

            m.city,
            m.gender,
            m.age,
            m.registered_via,
            m.registration_year,
            m.registration_month

        FROM labelled_customers c

        LEFT JOIN cleaned_members m
            ON c.msno = m.msno
        """
    )

    # Validate final member feature table
    checks = con.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT msno) AS unique_customers,

            COUNT(*) FILTER (
                WHERE msno IS NULL
            ) AS null_ids,

            COUNT(*) FILTER (
                WHERE is_churn NOT IN (0, 1)
                   OR is_churn IS NULL
            ) AS invalid_targets,

            COUNT(*) FILTER (
                WHERE age IS NOT NULL
                  AND age NOT BETWEEN 5 AND 99
            ) AS invalid_age,

            COUNT(*) FILTER (
                WHERE registration_month IS NOT NULL
                  AND registration_month NOT BETWEEN 1 AND 12
            ) AS invalid_month

        FROM member_features
        """
    ).fetchone()

    if checks is None:
        raise ValueError("Member feature validation query returned no results.")

    print("\nMember feature validation")
    print(f"Rows: {checks[0]:,}")
    print(f"Unique customers: {checks[1]:,}")
    print(f"Null IDs: {checks[2]:,}")
    print(f"Invalid targets: {checks[3]:,}")
    print(f"Invalid ages: {checks[4]:,}")
    print(f"Invalid months: {checks[5]:,}")

    if checks[0] != cohort_count[0]:
      raise ValueError("Member feature row count differs from cohort.")

    if checks[1] != cohort_count[1]:
      raise ValueError("Member feature output has duplicate customers.")

    if any(checks[i] > 0 for i in range(2, 6)):
      raise ValueError("Member feature validation failed.")

    # Save output
    con.execute(
        f"""
        COPY member_features
        TO '{OUTPUT_FILE.as_posix()}'
        (FORMAT PARQUET, COMPRESSION SNAPPY)
        """
    )

    print(f"\nSaved: {OUTPUT_FILE}")
    print("Member feature engineering completed.")

  finally:
    con.close()


if __name__ == "__main__":
  main()