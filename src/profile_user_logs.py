from pathlib import Path
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARQUET_GLOB = (
    PROJECT_ROOT
    / "data"
    / "kkbox"
    / "processed"
    / "log_chunks"
    / "user_logs_*.parquet"
)


def fetch_one_value(query: str):
    result = conn.execute(query).fetchone()
    return None if result is None else result[0]


conn = duckdb.connect()

print("\n=== USER LOG SCHEMA ===")

schema = conn.execute(
    f"""
    DESCRIBE
    SELECT *
    FROM read_parquet('{PARQUET_GLOB}')
    """
).fetchdf()

print(schema.to_string(index=False))


print("\n=== ROW COUNT ===")

row_count = fetch_one_value(
    f"""
    SELECT COUNT(*) AS row_count
    FROM read_parquet('{PARQUET_GLOB}')
    """
)

print(f"{row_count:,}" if row_count is not None else "0")


print("\n=== UNIQUE CUSTOMERS ===")

unique_customers = fetch_one_value(
    f"""
    SELECT COUNT(DISTINCT msno) AS unique_customers
    FROM read_parquet('{PARQUET_GLOB}')
    """
)

print(f"{unique_customers:,}" if unique_customers is not None else "0")


print("\n=== DATE RANGE ===")

date_range = conn.execute(
    f"""
    SELECT
        MIN(date) AS min_date,
        MAX(date) AS max_date
    FROM read_parquet('{PARQUET_GLOB}')
    """
).fetchone() or (None, None)

print(f"Minimum date: {date_range[0]}")
print(f"Maximum date: {date_range[1]}")


print("\n=== NULL COUNTS ===")

null_counts = conn.execute(
    f"""
    SELECT
        COUNT(*) FILTER (WHERE msno IS NULL) AS msno_nulls,
        COUNT(*) FILTER (WHERE date IS NULL) AS date_nulls,
        COUNT(*) FILTER (WHERE num_25 IS NULL) AS num_25_nulls,
        COUNT(*) FILTER (WHERE num_50 IS NULL) AS num_50_nulls,
        COUNT(*) FILTER (WHERE num_75 IS NULL) AS num_75_nulls,
        COUNT(*) FILTER (WHERE num_985 IS NULL) AS num_985_nulls,
        COUNT(*) FILTER (WHERE num_100 IS NULL) AS num_100_nulls,
        COUNT(*) FILTER (WHERE num_unq IS NULL) AS num_unq_nulls,
        COUNT(*) FILTER (WHERE total_secs IS NULL) AS total_secs_nulls
    FROM read_parquet('{PARQUET_GLOB}')
    """
).fetchone() or (0,) * 9

columns = [
    "msno",
    "date",
    "num_25",
    "num_50",
    "num_75",
    "num_985",
    "num_100",
    "num_unq",
    "total_secs",
]

for column, count in zip(columns, null_counts):
    print(f"{column}: {count:,}")


print("\n=== SAMPLE ===")

sample = conn.execute(
    f"""
    SELECT *
    FROM read_parquet('{PARQUET_GLOB}')
    LIMIT 5
    """
).fetchdf()

print(sample.to_string(index=False))


conn.close()