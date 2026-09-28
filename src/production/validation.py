"""Data quality checks that must pass before preprocessing/training."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_loader import DataLoader


REQUIRED_COLUMNS = {"time", "temperature", "humidity", "precipitation", "cloud_cover", "pressure_msl", "wind_speed_10m", "shortwave_radiation"}


def validate_dataset(data_path: Path, report_path: Path) -> Path:
    frame = DataLoader(data_path).load()
    checks = {
        "required_columns": REQUIRED_COLUMNS.issubset(frame.columns),
        "time_parseable": bool(frame["time"].notna().all()),
        "time_unique": bool(frame["time"].is_unique),
        "time_sorted": bool(frame["time"].is_monotonic_increasing),
        "numeric_values_finite": bool(frame.select_dtypes("number").notna().all().all()),
        "humidity_range": bool(frame["humidity"].between(0, 100).all()),
        "cloud_cover_range": bool(frame["cloud_cover"].between(0, 100).all()),
        "non_negative_precipitation": bool((frame["precipitation"] >= 0).all()),
    }
    result = {"data_path": str(data_path), "rows": int(len(frame)), "checks": checks, "passed": all(checks.values())}
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"data validation failed: {failed}")
    return report_path
