from pathlib import Path
import subprocess

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARCHIVE = (
    PROJECT_ROOT
    / "data"
    / "kkbox"
    / "raw"
    / "user_logs.csv.7z"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "kkbox"
    / "processed"
    / "log_chunks"
)

ROWS_PER_CHUNK = 500_000


DTYPES = {
    "msno": "string",
    "date": "string",
    "num_25": "int64",
    "num_50": "int64",
    "num_75": "int64",
    "num_985": "int64",
    "num_100": "int64",
    "num_unq": "int64",
    "total_secs": "float64",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        "7z",
        "e",
        "-so",
        str(ARCHIVE),
        "user_logs.csv",
    ]

    print("Starting streamed extraction from 7-Zip...")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    chunk_number = 0
    total_rows = 0
    stdout = process.stdout

    if stdout is None:
        raise RuntimeError("Failed to open 7-Zip stdout stream.")

    try:
        reader = pd.read_csv(
            stdout,
            dtype=DTYPES,
            chunksize=ROWS_PER_CHUNK,
        )

        for chunk in reader:
            table = pa.Table.from_pandas(
                chunk,
                preserve_index=False,
            )

            output_file = (
                OUTPUT_DIR
                / f"user_logs_{chunk_number:05d}.parquet"
            )

            pq.write_table(
                table,
                output_file,
                compression="snappy",
            )

            rows = len(chunk)
            total_rows += rows

            print(
                f"Wrote {output_file.name}: "
                f"{rows:,} rows | "
                f"total: {total_rows:,}"
            )

            chunk_number += 1

    finally:
        if process.stdout is not None:
            process.stdout.close()

    return_code = process.wait()

    if return_code != 0:
        stderr = process.stderr
        if stderr is None:
            raise RuntimeError(
                "7-Zip extraction failed, but no stderr output was captured."
            )

        error = stderr.read().decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            "7-Zip extraction failed:\n" + error
        )

    print()
    print("User-log conversion completed.")
    print(f"Total rows: {total_rows:,}")
    print(f"Parquet chunks: {chunk_number:,}")


if __name__ == "__main__":
    main()