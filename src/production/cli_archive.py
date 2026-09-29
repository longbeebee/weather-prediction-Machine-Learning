from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.production.storage import upload_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive MLOps data and artifacts to S3")
    parser.add_argument("--path", type=Path, action="append", required=True)
    parser.add_argument("--s3-bucket", default=os.getenv("S3_STORAGE_BUCKET"))
    parser.add_argument("--s3-prefix", required=True)
    parser.add_argument("--region", default=os.getenv("AWS_DEFAULT_REGION", ""))
    args = parser.parse_args()
    uploaded = upload_paths(args.path, args.s3_bucket, args.s3_prefix, args.region)
    print(f"uploaded_s3_objects={len(uploaded)}")
    for uri in uploaded:
        print(uri)


if __name__ == "__main__":
    main()
