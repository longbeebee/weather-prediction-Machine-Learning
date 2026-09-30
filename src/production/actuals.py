"""Collect historical weather outcomes for production forecast evaluation."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _forecast_times(predictions_jsonl: Path) -> pd.DatetimeIndex:
    values = []
    for line in Path(predictions_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        values.extend(item["forecast_time"] for item in payload.get("forecast", []))
    if not values:
        raise ValueError("prediction log contains no forecast_time values")
    return pd.DatetimeIndex(pd.to_datetime(values, errors="coerce")).dropna().unique().sort_values()


def collect_actuals(
    predictions_jsonl: Path,
    output_csv: Path,
    metadata_path: Path,
    latitude: float,
    longitude: float,
    timezone: str = "Asia/Bangkok",
    now: datetime | None = None,
) -> Path:
    """Fetch actual weather for forecast timestamps that have already occurred."""
    times = _forecast_times(predictions_jsonl)
    current = pd.Timestamp(now or datetime.now(ZoneInfo(timezone))).tz_localize(None)
    occurred = times[times <= current]
    if len(occurred) == 0:
        raise ValueError("no forecast timestamps have occurred yet; wait until a forecast is verifiable")

    start_date = occurred.min().strftime("%Y-%m-%d")
    end_date = occurred.max().strftime("%Y-%m-%d")
    query = urllib.parse.urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation",
        "timezone": timezone,
    })
    request_url = f"{OPEN_METEO_ARCHIVE_URL}?{query}"
    request = urllib.request.Request(request_url, headers={"User-Agent": "weather-7d-mlops-demo/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason', payload['error'])}")

    hourly = payload.get("hourly", {})
    actuals = pd.DataFrame({
        "forecast_time": hourly.get("time", []),
        "actual_temperature": hourly.get("temperature_2m", []),
        "actual_precipitation": hourly.get("precipitation", []),
    })
    if actuals.empty:
        raise ValueError("Open-Meteo returned no hourly actual data")
    actuals["forecast_time"] = pd.to_datetime(actuals["forecast_time"], errors="coerce")
    actuals["actual_temperature"] = pd.to_numeric(actuals["actual_temperature"], errors="coerce")
    actuals["actual_precipitation"] = pd.to_numeric(actuals["actual_precipitation"], errors="coerce")
    actuals["actual_rain"] = actuals["actual_precipitation"].fillna(0) > 0
    actuals = actuals.dropna(subset=["forecast_time", "actual_temperature"])
    actuals = actuals[actuals["forecast_time"].isin(occurred)].copy()
    if actuals.empty:
        raise ValueError("Open-Meteo returned no rows matching production forecast_time values")

    output_csv = Path(output_csv)
    metadata_path = Path(metadata_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    actuals.to_csv(output_csv, index=False)
    metadata_path.write_text(json.dumps({
        "source": "Open-Meteo Historical Weather API",
        "source_url": request_url,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "start_date": start_date,
        "end_date": end_date,
        "rows": len(actuals),
        "forecast_times": [str(value) for value in occurred],
    }, indent=2), encoding="utf-8")
    return output_csv
