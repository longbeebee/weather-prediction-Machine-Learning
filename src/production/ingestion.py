"""Idempotent ingestion step used by the Airflow training DAG."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.config import FALLBACK_RAW_DATA_PATH, RAW_DATA_PATH


def ingest(output_path: Path, source_path: Path | None = None) -> Path:
    source = source_path or (RAW_DATA_PATH if RAW_DATA_PATH.exists() else FALLBACK_RAW_DATA_PATH)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)
    return output_path
