from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.production.storage import build_feature_snapshot, upload_feature_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and optionally upload a versioned feature snapshot")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--s3-bucket", default=os.getenv("S3_FEATURE_BUCKET"))
    parser.add_argument("--s3-prefix", default=os.getenv("S3_FEATURE_PREFIX", "weather-7d/features"))
    parser.add_argument("--region", default=os.getenv("AWS_DEFAULT_REGION", ""))
    args = parser.parse_args()
    metadata = build_feature_snapshot(args.data_path, args.output_path, args.metadata_path)
    if not args.s3_bucket:
        raise ValueError("--s3-bucket or S3_FEATURE_BUCKET is required")
    print(upload_feature_snapshot(metadata, args.output_path, args.s3_bucket, args.s3_prefix, args.region))


if __name__ == "__main__":
    main()
