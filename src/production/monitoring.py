"""Performance, drift, and retraining decision utilities."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error


def _read_dataset(path: Path) -> pd.DataFrame:
    """Read the lake format without requiring pyarrow in Airflow images."""
    path = Path(path)
    if path.suffix != ".parquet":
        return pd.read_csv(path)
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("DuckDB is required to read the monitoring Parquet dataset") from exc
    with duckdb.connect() as connection:
        return connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetch_df()


def run_evidently_drift_report(data_path: Path, html_path: Path, json_path: Path, current_features_path: Path | None = None, current_rows: int = 30, reference_rows: int = 180) -> Path:
    """Compare current production-input features with a historical reference window."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("Evidently is required for drift monitoring") from exc

    from src.config import FALLBACK_RAW_DATA_PATH, RAW_DATA_PATH
    from src.data_loader import DataLoader
    from src.preprocessing import Preprocessor
    from src.production.features import build_base_features, select_feature_columns

    source = Path(data_path)
    if not source.exists():
        source = RAW_DATA_PATH if RAW_DATA_PATH.exists() else FALLBACK_RAW_DATA_PATH
    raw = DataLoader(source).load()
    noon = Preprocessor().preprocess(raw, save=False)
    features = build_base_features(noon)
    columns = select_feature_columns(features)
    if len(features) < current_rows + reference_rows:
        raise ValueError("not enough rows for Evidently reference/current windows")
    reference = features.iloc[-(current_rows + reference_rows):-current_rows][columns].copy()
    if current_features_path:
        log_path = Path(current_features_path)
        if not log_path.exists():
            raise FileNotFoundError(log_path)
        if log_path.suffix == ".parquet":
            current = _read_dataset(log_path)
            # One API request produces seven horizon rows but only one online
            # feature snapshot. Drift must count requests, not horizons.
            if "request_id" in current.columns:
                current = current.drop_duplicates("request_id", keep="last")
        else:
            records = []
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    payload = json.loads(line)
                    if payload.get("feature_values"):
                        records.append(payload["feature_values"])
            current = pd.DataFrame(records)
        if len(current) < current_rows:
            raise ValueError(f"need at least {current_rows} production feature logs, found {len(current)}")
        missing = [column for column in columns if column not in current.columns]
        if missing:
            raise ValueError(f"production feature log is missing columns: {missing}")
        current = current[columns].apply(pd.to_numeric, errors="coerce").dropna()
    else:
        current = features.iloc[-current_rows:][columns].copy()

    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)
    html_path = Path(html_path)
    json_path = Path(json_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    payload = snapshot.dict()
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def find_values(value: object, key: str) -> list[object]:
        found: list[object] = []
        if isinstance(value, dict):
            if key in value:
                found.append(value[key])
            for child in value.values():
                found.extend(find_values(child, key))
        elif isinstance(value, list):
            for child in value:
                found.extend(find_values(child, key))
        return found

    # Evidently report schemas differ between versions.  Depending on the
    # version, the UI's dataset-level result is exposed as
    # ``dataset_drift`` or as a drifted-column share.  Read both so that the
    # machine-readable summary cannot disagree with the HTML report.
    drift_values = find_values(payload, "dataset_drift")
    drifted_counts = find_values(payload, "number_of_drifted_columns")
    # Evidently 0.7 serializes this metric as:
    # {"metric_name": "DriftedColumnsCount(...) ",
    #  "config": {"drift_share": 0.5},
    #  "value": {"count": 20.0, "share": 0.8}}
    # The configured threshold and the observed result are intentionally
    # separate fields.
    drift_metric_shares: list[float] = []
    drift_metric_counts: list[int] = []
    for metric in payload.get("metrics", []) if isinstance(payload, dict) else []:
        if not isinstance(metric, dict) or "DriftedColumnsCount" not in str(metric.get("metric_name", "")):
            continue
        metric_value = metric.get("value")
        if isinstance(metric_value, dict):
            share = metric_value.get("share")
            count = metric_value.get("count")
            if isinstance(share, (int, float)):
                drift_metric_shares.append(float(share))
            if isinstance(count, (int, float)):
                drift_metric_counts.append(int(count))
    # ``drift_share`` is also used by Evidently as the configured decision
    # threshold (commonly 0.5).  It must not be treated as the observed share
    # shown in the UI.  Prefer result fields that explicitly describe the
    # measured share of drifted columns/features.
    drift_shares = drift_metric_shares + (
        find_values(payload, "share_of_drifted_columns")
        + find_values(payload, "share_of_drifted_features")
        + find_values(payload, "drifted_columns_share")
    )

    numeric_shares = [float(value) for value in drift_shares if isinstance(value, (int, float))]
    drift_share = max(numeric_shares, default=0.0)
    if drift_share > 1.0:
        drift_share /= 100.0
    drifted_count = max(
        drift_metric_counts
        + [int(value) for value in drifted_counts if isinstance(value, (int, float))],
        default=0,
    )
    explicit_drift = any(value is True for value in drift_values)
    drift_detected = explicit_drift or drifted_count > 0 or drift_share > 0.0
    summary = {
        "tool": "evidently",
        "reference_rows": len(reference),
        "current_rows": len(current),
        "feature_count": len(columns),
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "drifted_feature_count": drifted_count,
        "html_report": str(html_path),
        "json_report": str(json_path),
    }
    summary_path = json_path.with_name("evidently_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def apply_drift_result(report_path: Path, drift_summary_path: Path, decision_path: Path, pushgateway_url: str | None = None) -> Path:
    """Merge Evidently drift into the monitoring decision and publish final metrics."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    drift = json.loads(Path(drift_summary_path).read_text(encoding="utf-8"))
    report["drift_detected"] = bool(drift["drift_detected"])
    report["drifted_feature_count"] = int(drift.get("drifted_feature_count", 0))
    report["retraining_required"] = bool(report.get("performance_degraded") or report["drift_detected"])
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    decide_retraining(Path(report_path), Path(decision_path))
    if pushgateway_url:
        push_monitoring_metrics(Path(report_path), pushgateway_url)
    return Path(report_path)


def monitor_predictions(dataset_path: Path, output_path: Path, rmse_limit: float = 4.0, f1_limit: float = 0.55) -> Path:
    """Evaluate the canonical materialized prediction/actual dataset."""
    joined = _read_dataset(dataset_path)
    required = {"horizon_day", "predicted_temperature", "predicted_rain", "actual_temperature", "actual_rain"}
    if not required.issubset(joined.columns):
        raise ValueError(f"monitoring dataset is missing columns: {sorted(required - set(joined.columns))}")
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
        f"weather_model_drifted_feature_count {int(report.get('drifted_feature_count', 0))}",
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
