from __future__ import annotations

import argparse

from src.production.query import query_s3_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Query an S3 Parquet dataset with DuckDB")
    parser.add_argument("--path", required=True, help="s3://.../*.parquet")
    parser.add_argument("--sql", default="SELECT * FROM lake_data LIMIT 20")
    parser.add_argument("--region", default=None)
    args = parser.parse_args()
    print(query_s3_parquet(args.path, args.sql, args.region).to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
