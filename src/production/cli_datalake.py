from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.production.datalake import materialize_monitoring_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize monitoring data as Parquet on S3")
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--actuals-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--s3-bucket", default=os.getenv("S3_STORAGE_BUCKET"))
    parser.add_argument("--s3-prefix", default="weather-7d/bronze")
    parser.add_argument("--region", default=os.getenv("AWS_DEFAULT_REGION", ""))
    parser.add_argument("--release-id", default=None)
    args = parser.parse_args()
    if not args.s3_bucket:
        raise ValueError("--s3-bucket or S3_STORAGE_BUCKET is required")
    print(materialize_monitoring_data(args.predictions_jsonl, args.actuals_csv, args.output_dir, args.s3_bucket, args.s3_prefix, args.region, args.release_id))


if __name__ == "__main__":
    main()
