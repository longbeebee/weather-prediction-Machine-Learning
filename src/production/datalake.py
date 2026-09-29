"""Batch materialization of API logs and actuals into S3 Parquet data lake files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.production.storage import upload_paths, write_parquet, write_s3_catalog


def prediction_log_frame(path: Path) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        feature_values = payload.get("feature_values", {})
        for forecast in payload.get("forecast", []):
            row = {
                "request_id": payload["request_id"],
                "observed_at": payload["observed_at"],
                "contract_version": payload["contract_version"],
                "model_created_at": payload["model_created_at"],
                "observation_time": payload["observation_time"],
                "horizon_day": forecast["horizon_day"],
                "forecast_time": forecast["forecast_time"],
                "predicted_temperature": forecast["predicted_temperature"],
                "predicted_rain": forecast["predicted_rain"],
                "predicted_rain_probability": forecast["predicted_rain_probability"],
            }
            # Keep the exact online feature snapshot in the lake. This makes
            # the same materialized dataset usable by both performance and
            # Evidently drift monitoring.
            row.update(feature_values)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["forecast_time"] = pd.to_datetime(frame["forecast_time"], errors="coerce")
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], errors="coerce")
        frame["observation_time"] = pd.to_datetime(frame["observation_time"], errors="coerce")
    return frame


def normalize_actuals(actuals: pd.DataFrame) -> pd.DataFrame:
    """Normalize observed weather into the prediction join contract."""
    frame = actuals.copy()
    aliases = {
        "actual_time": "forecast_time",
        "time": "forecast_time",
        "actual_temperature": "actual_temperature",
        "temperature": "actual_temperature",
        "temperature_2m (°C)": "actual_temperature",
        "temperature_2m (degC)": "actual_temperature",
        "actual_rain": "actual_rain",
        "rain": "actual_rain",
        "precipitation": "actual_precipitation",
        "precipitation (mm)": "actual_precipitation",
    }
    frame = frame.rename(columns={column: aliases.get(column, column) for column in frame.columns})
    required = {"forecast_time", "actual_temperature"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"actuals are missing columns: {sorted(missing)}")
    if "actual_rain" not in frame.columns:
        if "actual_precipitation" not in frame.columns:
            raise ValueError("actuals need actual_rain/rain or precipitation")
        frame["actual_rain"] = pd.to_numeric(frame["actual_precipitation"], errors="coerce") > 0
    frame["forecast_time"] = pd.to_datetime(frame["forecast_time"], errors="coerce")
    frame["actual_temperature"] = pd.to_numeric(frame["actual_temperature"], errors="coerce")
    if frame["actual_rain"].dtype == object:
        frame["actual_rain"] = frame["actual_rain"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "rain"})
    else:
        frame["actual_rain"] = pd.to_numeric(frame["actual_rain"], errors="coerce").fillna(0) > 0
    frame = frame.dropna(subset=["forecast_time", "actual_temperature"])
    return frame[["forecast_time", "actual_temperature", "actual_rain"]].drop_duplicates("forecast_time")


def materialize_monitoring_data(predictions_jsonl: Path, actuals_csv: Path, output_dir: Path, bucket: str, prefix: str, region: str | None = None) -> dict:
    output_dir = Path(output_dir)
    predictions = prediction_log_frame(predictions_jsonl)
    if predictions.empty:
        raise ValueError("prediction log is empty; call the production API before materializing the data lake")
    actuals = normalize_actuals(pd.read_csv(actuals_csv))
    predictions["forecast_time"] = pd.to_datetime(predictions["forecast_time"], errors="coerce")
    predictions = predictions.dropna(subset=["forecast_time"])
    if not predictions.empty:
        predictions["event_date"] = pd.to_datetime(predictions["observed_at"]).dt.strftime("%Y-%m-%d")
    actuals["event_date"] = actuals["forecast_time"].dt.strftime("%Y-%m-%d")
    joined = predictions.merge(actuals, on="forecast_time", how="inner", validate="many_to_one")
    if joined.empty:
        raise ValueError("no production predictions matched actual observations by forecast_time")
    joined["event_date"] = joined["forecast_time"].dt.strftime("%Y-%m-%d")
    prediction_path = write_parquet(predictions, output_dir / "predictions.parquet")
    actual_path = write_parquet(actuals, output_dir / "actuals.parquet")
    dataset_path = write_parquet(joined, output_dir / "monitoring_dataset.parquet")
    upload_paths([prediction_path], bucket, f"{prefix.strip('/')}/predictions", region)
    upload_paths([actual_path], bucket, f"{prefix.strip('/')}/actuals", region)
    upload_paths([dataset_path], bucket, f"{prefix.strip('/')}/dataset", region)
    catalog = write_s3_catalog(
        output_dir / "datasets.json",
        bucket,
        f"{prefix.strip('/')}/_catalog",
        {
            "predictions": {"format": "parquet", "uri": f"s3://{bucket}/{prefix.strip('/')}/predictions/predictions.parquet"},
            "actuals": {"format": "parquet", "uri": f"s3://{bucket}/{prefix.strip('/')}/actuals/actuals.parquet"},
            "monitoring_dataset": {"format": "parquet", "uri": f"s3://{bucket}/{prefix.strip('/')}/dataset/monitoring_dataset.parquet"},
        },
        region,
    )
    return {"predictions": str(prediction_path), "actuals": str(actual_path), "monitoring_dataset": str(dataset_path), "catalog": str(catalog)}
