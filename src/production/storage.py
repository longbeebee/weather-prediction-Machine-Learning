"""Versioned feature snapshot storage on S3."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.production.features import build_base_features, select_feature_columns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(frame: pd.DataFrame, output_path: Path) -> Path:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("duckdb is required to write Parquet data lake files") from exc
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.register("source_frame", frame)
        connection.execute("COPY source_frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(output_path)])
    finally:
        connection.close()
    return output_path


def build_feature_snapshot(data_path: Path, output_path: Path, metadata_path: Path) -> Path:
    raw = DataLoader(data_path).load()
    noon = Preprocessor().preprocess(raw, save=False)
    features = build_base_features(noon)
    feature_columns = select_feature_columns(features)
    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.with_suffix(".parquet")
    write_parquet(features, output_path)
    version = _sha256(output_path)[:16]
    metadata = {
        "feature_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_data_path": str(data_path),
        "source_data_sha256": _sha256(Path(data_path)),
        "rows": int(len(features)),
        "feature_columns": feature_columns,
        "format": "parquet",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def upload_feature_snapshot(metadata_path: Path, feature_path: Path, bucket: str, prefix: str, region: str | None = None) -> Path:
    if not bucket:
        raise ValueError("S3_FEATURE_BUCKET is required for feature storage")
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("boto3 is required for S3 feature storage") from exc
    metadata_path = Path(metadata_path)
    feature_path = Path(feature_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    version = metadata["feature_version"]
    clean_prefix = prefix.strip("/")
    base_key = f"{clean_prefix}/feature_version={version}" if clean_prefix else f"feature_version={version}"
    client = boto3.client("s3", region_name=region or None)
    client.upload_file(str(feature_path), bucket, f"{base_key}/features.parquet")
    metadata.update({
        "s3_bucket": bucket,
        "s3_prefix": base_key,
        "s3_features_uri": f"s3://{bucket}/{base_key}/features.parquet",
        "s3_metadata_uri": f"s3://{bucket}/{base_key}/metadata.json",
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    client.upload_file(str(metadata_path), bucket, f"{base_key}/metadata.json")
    return metadata_path


def upload_paths(paths: list[Path], bucket: str, prefix: str, region: str | None = None) -> list[str]:
    """Upload files/directories to an immutable S3 artifact prefix."""
    if not bucket:
        raise ValueError("S3_STORAGE_BUCKET is required for artifact storage")
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("boto3 is required for S3 artifact storage") from exc
    client = boto3.client("s3", region_name=region or None)
    uploaded: list[str] = []
    root_prefix = prefix.strip("/")
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
        for file_path in files:
            relative = file_path.name if path.is_file() else str(file_path.relative_to(path.parent)).replace("\\", "/")
            key = f"{root_prefix}/{relative}" if root_prefix else relative
            client.upload_file(str(file_path), bucket, key, ExtraArgs={"Metadata": {"sha256": _sha256(file_path)}})
            uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def load_feature_snapshot(path: Path) -> pd.DataFrame:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("duckdb is required to read Parquet feature snapshots") from exc
    connection = duckdb.connect()
    try:
        frame = connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()
    finally:
        connection.close()
    frame["time"] = pd.to_datetime(frame["time"])
    return frame.sort_values("time").reset_index(drop=True)


def write_s3_catalog(catalog_path: Path, bucket: str, prefix: str, datasets: dict, region: str | None = None) -> Path:
    catalog_path = Path(catalog_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"catalog_version": "v1", "updated_at": datetime.now(timezone.utc).isoformat(), "datasets": datasets}
    catalog_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    upload_paths([catalog_path], bucket, prefix, region)
    return catalog_path
