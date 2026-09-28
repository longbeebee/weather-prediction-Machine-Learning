"""Performance, drift, and retraining decision utilities."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error


def monitor_predictions(predictions_path: Path, actuals_path: Path, output_path: Path, rmse_limit: float = 4.0, f1_limit: float = 0.55) -> Path:
    predictions = pd.read_csv(predictions_path)
    actuals = pd.read_csv(actuals_path)
    required_prediction = {"horizon_day", "predicted_temperature", "predicted_rain"}
    required_actual = {"horizon_day", "actual_temperature", "actual_rain"}
    if not required_prediction.issubset(predictions.columns) or not required_actual.issubset(actuals.columns):
        raise ValueError("predictions and actuals must contain horizon and target columns")
    joined = predictions.merge(actuals, on="horizon_day", how="inner")
    rows = []
    for horizon, frame in joined.groupby("horizon_day"):
        rmse = float(np.sqrt(mean_squared_error(frame["actual_temperature"], frame["predicted_temperature"])))
        f1 = float(f1_score(frame["actual_rain"], frame["predicted_rain"], zero_division=0))
        rows.append({"horizon_day": int(horizon), "rmse": rmse, "mae": float(mean_absolute_error(frame["actual_temperature"], frame["predicted_temperature"])), "rain_f1": f1, "performance_degraded": rmse > rmse_limit or f1 < f1_limit})
    result = {"rows": rows, "performance_degraded": any(row["performance_degraded"] for row in rows), "drift_detected": False, "retraining_required": any(row["performance_degraded"] for row in rows)}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path


def decide_retraining(monitoring_report: Path, output_path: Path) -> Path:
    report = json.loads(Path(monitoring_report).read_text(encoding="utf-8"))
    decision = {"retrain": bool(report.get("retraining_required") or report.get("drift_detected")), "reasons": [], "source_report": str(monitoring_report)}
    if report.get("performance_degraded"):
        decision["reasons"].append("performance_degraded")
    if report.get("drift_detected"):
        decision["reasons"].append("data_drift_detected")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return output_path


def push_monitoring_metrics(report_path: Path, pushgateway_url: str, job: str = "weather_7d_monitoring") -> None:
    """Push bounded-label model metrics for short-lived Airflow jobs."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    lines = [
        f"weather_model_performance_degraded {int(bool(report.get('performance_degraded')))}",
        f"weather_model_drift_detected {int(bool(report.get('drift_detected')))}",
        f"weather_model_retraining_required {int(bool(report.get('retraining_required')))}",
    ]
    for row in report.get("rows", []):
        horizon = int(row["horizon_day"])
        lines.extend([
            f'weather_model_rmse{{horizon="{horizon}"}} {float(row["rmse"])}',
            f'weather_model_mae{{horizon="{horizon}"}} {float(row["mae"])}',
            f'weather_model_rain_f1{{horizon="{horizon}"}} {float(row["rain_f1"])}',
        ])
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    url = pushgateway_url.rstrip("/") + "/metrics/job/" + job
    request = urllib.request.Request(url, data=payload, method="PUT", headers={"Content-Type": "text/plain; version=0.0.4"})
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status >= 300:
            raise RuntimeError(f"Pushgateway returned HTTP {response.status}")
