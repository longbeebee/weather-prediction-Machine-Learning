import json

import pandas as pd

from src.production.datalake import normalize_actuals, prediction_log_frame
from src.production.monitoring import monitor_predictions


def test_prediction_log_and_actuals_share_forecast_time_contract(tmp_path):
    log = tmp_path / "api_predictions.jsonl"
    log.write_text(json.dumps({
        "request_id": "request-1",
        "observed_at": "2026-01-01T12:00:00+00:00",
        "contract_version": "weather-7d-v1",
        "model_created_at": "2026-01-01T00:00:00+00:00",
        "observation_time": "2026-01-01T12:00:00",
        "feature_values": {"temperature": 20.0},
        "forecast": [{
            "horizon_day": 1,
            "forecast_time": "2026-01-02T12:00:00",
            "predicted_temperature": 21.0,
            "predicted_rain": False,
            "predicted_rain_probability": 0.1,
        }],
    }) + "\n", encoding="utf-8")

    predictions = prediction_log_frame(log)
    actuals = normalize_actuals(pd.DataFrame({
        "time": ["2026-01-02T12:00:00"],
        "temperature": [20.5],
        "precipitation": [0.0],
    }))

    joined = predictions.merge(actuals, on="forecast_time", validate="many_to_one")
    assert len(joined) == 1
    assert joined.loc[0, "actual_temperature"] == 20.5
    assert bool(joined.loc[0, "actual_rain"]) is False
    assert joined.loc[0, "temperature"] == 20.0


def test_monitoring_uses_canonical_joined_dataset(tmp_path):
    dataset = tmp_path / "monitoring_dataset.csv"
    pd.DataFrame([
        {"horizon_day": 1, "predicted_temperature": 21.0, "predicted_rain": False, "actual_temperature": 20.5, "actual_rain": False},
        {"horizon_day": 1, "predicted_temperature": 22.0, "predicted_rain": True, "actual_temperature": 22.0, "actual_rain": True},
    ]).to_csv(dataset, index=False)
    report = tmp_path / "performance_report.json"

    monitor_predictions(dataset, report)

    payload = report.read_text(encoding="utf-8")
    assert '"horizon_day": 1' in payload
    assert '"performance_degraded": false' in payload
