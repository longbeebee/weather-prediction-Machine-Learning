"""DuckDB query helpers for Parquet data stored on S3."""

from __future__ import annotations

import os

import duckdb


def query_s3_parquet(uri: str, sql: str, region: str | None = None):
    connection = duckdb.connect()
    try:
        connection.execute("INSTALL httpfs")
        connection.execute("LOAD httpfs")
        connection.execute("INSTALL aws")
        connection.execute("LOAD aws")
        selected_region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        connection.execute(f"CREATE OR REPLACE SECRET weather_s3 (TYPE s3, PROVIDER credential_chain, REGION '{selected_region}')")
        connection.execute("CREATE OR REPLACE VIEW lake_data AS SELECT * FROM read_parquet(?)", [uri])
        return connection.execute(sql).df()
    finally:
        connection.close()
