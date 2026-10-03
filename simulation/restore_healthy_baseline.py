"""Call the prediction API with a healthy historical baseline.

The script appends fresh baseline requests to the API's configured prediction
log. It does not delete old logs. The generated monitoring fixture uses
bounded synthetic outcomes so the demo performance gate stays healthy; this
is not a replacement for production actual observations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

# Make ``python simulation/restore_healthy_baseline.py`` work from the
# repository root as documented, without requiring PYTHONPATH to be set.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.production.features import build_base_features, select_feature_columns
from src.production.monitoring import monitor_predictions, run_evidently_drift_report


def _build_request(history: pd.DataFrame, start: int) -> dict:
    """Convert eight historical noon rows into the API request contract."""
    columns = [
        "time", "temperature", "humidity", "precipitation", "cloud_cover",
        "pressure_msl", "wind_speed_10m", "shortwave_radiation",
    ]
    window = history.iloc[start:start + 8]
    return {"observations": window[columns].to_dict(orient="records")}


def _call_api(url: str, payload: dict, timeout: int) -> dict:
    request = Request(
        url,
        data=json.dumps(payload, default=str).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"prediction API returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"cannot reach prediction API at {url}: {exc.reason}") from exc


def _healthy_monitoring_dataset(responses: list[dict]) -> pd.DataFrame:
    """Build outcomes close to API predictions for the demo gate."""
    records: list[dict] = []
    for index, response in enumerate(responses):
        for forecast in response.get("forecast", []):
            predicted_temperature = float(forecast["predicted_temperature"])
            records.append({
                "request_id": response.get("request_id", f"healthy-{index + 1:03d}"),
                "model_release_id": response.get("model_release_id"),
                "horizon_day": int(forecast["horizon_day"]),
                "predicted_temperature": predicted_temperature,
                "actual_temperature": predicted_temperature + (0.15 if index % 2 else -0.15),
                "predicted_rain": bool(forecast["predicted_rain"]),
                "actual_rain": bool(forecast["predicted_rain"]),
            })
    return pd.DataFrame(records)


def _write_feature_log(responses: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for response in responses:
            stream.write(json.dumps({"model_release_id": response.get("model_release_id"), "feature_values": response["feature_values"]}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call the prediction API with a healthy baseline"
    )
    parser.add_argument("--url", default=os.getenv("PREDICTION_API_URL", "http://localhost:8000/api/v1/predict"))
    parser.add_argument("--data-path", type=Path, default=Path("data/processed/noon_weather_cleaned.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("simulation/healthy_baseline"))
    parser.add_argument("--count", type=int, default=35, help="Number of baseline API calls; use at least 30 for drift monitoring")
    parser.add_argument("--delay-seconds", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--reference-rows", type=int, default=180)
    args = parser.parse_args()

    if args.count < 30:
        raise ValueError("--count must be at least 30 for the default Evidently window")
    if args.delay_seconds < 0 or args.reference_rows <= 0:
        raise ValueError("--delay-seconds must be non-negative and --reference-rows must be greater than zero")

    raw = pd.read_csv(args.data_path)
    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    raw = raw.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    features = build_base_features(raw)
    columns = select_feature_columns(features)
    required = args.count + 8
    if len(raw) < required:
        raise ValueError(f"need at least {required} usable rows, found {len(raw)}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    responses: list[dict] = []
    response_path = output_dir / "prediction_responses.jsonl"
    with response_path.open("w", encoding="utf-8") as stream:
        for index in range(args.count):
            response = _call_api(args.url, _build_request(raw, index), args.timeout)
            if "feature_values" not in response or "forecast" not in response:
                raise ValueError("prediction API response is missing feature_values or forecast")
            responses.append(response)
            stream.write(json.dumps(response, default=str) + "\n")
            stream.flush()
            print(f"[{index + 1}/{args.count}] request_id={response.get('request_id')} observation_time={response.get('observation_time')}")
            if args.delay_seconds:
                time.sleep(args.delay_seconds)

    current_features_path = output_dir / "current_features.jsonl"
    _write_feature_log(responses, current_features_path)

    monitoring_dataset = _healthy_monitoring_dataset(responses)
    dataset_path = output_dir / "monitoring_dataset.csv"
    monitoring_dataset.to_csv(dataset_path, index=False)
    performance_path = output_dir / "performance_report.json"
    decision_path = output_dir / "retraining_decision.json"
    release_id = responses[0].get("model_release_id")
    monitor_predictions(dataset_path, performance_path, release_id=release_id)

    try:
        drift_summary_path = run_evidently_drift_report(
            args.data_path,
            output_dir / "drift_report.html",
            output_dir / "drift_report.json",
            current_features_path=current_features_path,
            current_rows=args.count,
            reference_rows=args.reference_rows,
            release_id=release_id,
        )
    except RuntimeError as exc:
        if "Evidently is required" not in str(exc):
            raise
        # Keep the offline fixture runnable before optional monitoring
        # dependencies are installed.  The generated current window is taken
        # from the same historical source, so this fallback is intentionally
        # a baseline result rather than a replacement for Evidently.
        drift_summary_path = output_dir / "evidently_summary.json"
        drift_summary_path.write_text(json.dumps({
            "tool": "baseline_fixture",
            "reference_rows": args.reference_rows,
            "current_rows": args.count,
            "feature_count": len(columns),
            "drift_detected": False,
            "drift_share": 0.0,
            "drifted_feature_count": 0,
            "warning": "Evidently is not installed; install requirements for a real drift report",
        }, indent=2), encoding="utf-8")
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    drift = json.loads(drift_summary_path.read_text(encoding="utf-8"))
    decision = {
        "retrain": bool(performance.get("performance_degraded") or drift.get("drift_detected")),
        "reasons": [],
        "source_report": str(performance_path),
        "simulation": "healthy_baseline",
    }
    if performance.get("performance_degraded"):
        decision["reasons"].append("performance_degraded")
    if drift.get("drift_detected"):
        decision["reasons"].append("data_drift_detected")
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    summary = {
        "simulation": "healthy_baseline",
        "data_path": str(args.data_path),
        "output_dir": str(output_dir),
        "performance_degraded": bool(performance.get("performance_degraded")),
        "drift_detected": bool(drift.get("drift_detected")),
        "retraining_required": bool(decision["retrain"]),
        "artifacts": {
            "monitoring_dataset": str(dataset_path),
            "current_features": str(current_features_path),
            "performance_report": str(performance_path),
            "drift_summary": str(drift_summary_path),
            "retraining_decision": str(decision_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
