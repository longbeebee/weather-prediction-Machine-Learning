"""Evaluation and promotion stage for the 7-day production contract."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, r2_score

from src.config import FALLBACK_RAW_DATA_PATH, RAW_DATA_PATH
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.production.features import build_base_features, build_horizon_targets, merge_horizon_targets
from src.production.training import CONTRACT_VERSION
from src.production.storage import load_feature_snapshot
from src.seven_day_candidates import rain_probabilities
from src.split import chronological_split


def evaluate_candidates(candidate_manifest: Path, output_dir: Path, data_path: Path | None = None, features_path: Path | None = None) -> Path:
    """Select using validation, score once on test, and publish a serving manifest."""
    candidate_manifest = Path(candidate_manifest)
    output_dir = Path(output_dir)
    manifest = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    if manifest.get("stage") != "candidate":
        raise ValueError("evaluation requires a candidate manifest")
    csv_path = data_path or (RAW_DATA_PATH if RAW_DATA_PATH.exists() else FALLBACK_RAW_DATA_PATH)
    raw = DataLoader(csv_path).load()
    noon = Preprocessor().preprocess(raw, save=False)
    base_features = load_feature_snapshot(features_path) if features_path else build_base_features(noon)
    feature_columns = manifest["feature_columns"]
    output_dir.mkdir(parents=True, exist_ok=True)

    horizons = []
    metrics = []
    for horizon_info in manifest["horizons"]:
        horizon = int(horizon_info["horizon_day"])
        frame = merge_horizon_targets(base_features, build_horizon_targets(noon, horizon))
        _, validation, test = chronological_split(frame)
        candidates = {
            "temperature": {name: joblib.load(candidate_manifest.parent / path) for name, path in horizon_info["temperature_models"].items()},
            "rain": {name: joblib.load(candidate_manifest.parent / path) for name, path in horizon_info["rain_models"].items()},
        }
        X_val, X_test = validation[feature_columns], test[feature_columns]
        temp_scores = []
        for name, model in candidates["temperature"].items():
            prediction = model.predict(X_val)
            temp_scores.append((name, float(np.sqrt(mean_squared_error(validation["horizon_temperature"], prediction)))))
        best_temp = min(temp_scores, key=lambda item: item[1])[0]

        rain_scores = []
        thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
        rain_threshold = 0.5
        for name, model in candidates["rain"].items():
            probabilities = rain_probabilities(model, X_val)
            best_f1 = -1.0
            best_threshold = 0.5
            for threshold in thresholds:
                score = f1_score(validation["horizon_rain"].astype(int), (probabilities >= threshold).astype(int), zero_division=0)
                if score > best_f1:
                    best_f1, best_threshold = float(score), float(threshold)
            rain_scores.append((name, best_f1, best_threshold))
        best_rain, _, rain_threshold = max(rain_scores, key=lambda item: item[1])

        validation_temp = candidates["temperature"][best_temp].predict(X_val)
        validation_rain_prob = rain_probabilities(candidates["rain"][best_rain], X_val)
        validation_rain = (validation_rain_prob >= rain_threshold).astype(int)
        metrics.extend([
            {"horizon_day": horizon, "split": "validation", "task": "temperature_regression", "model": best_temp, "mae": mean_absolute_error(validation["horizon_temperature"], validation_temp), "rmse": np.sqrt(mean_squared_error(validation["horizon_temperature"], validation_temp)), "r2": r2_score(validation["horizon_temperature"], validation_temp)},
            {"horizon_day": horizon, "split": "validation", "task": "rain_classification", "model": best_rain, "threshold": rain_threshold, "accuracy": accuracy_score(validation["horizon_rain"], validation_rain), "precision": precision_score(validation["horizon_rain"], validation_rain, zero_division=0), "recall": recall_score(validation["horizon_rain"], validation_rain, zero_division=0), "f1": f1_score(validation["horizon_rain"], validation_rain, zero_division=0)},
        ])

        temp_test = candidates["temperature"][best_temp].predict(X_test)
        rain_prob = rain_probabilities(candidates["rain"][best_rain], X_test)
        rain_test = (rain_prob >= rain_threshold).astype(int)
        metrics.extend([
            {"horizon_day": horizon, "split": "test", "task": "temperature_regression", "model": best_temp, "mae": mean_absolute_error(test["horizon_temperature"], temp_test), "rmse": np.sqrt(mean_squared_error(test["horizon_temperature"], temp_test)), "r2": r2_score(test["horizon_temperature"], temp_test)},
            {"horizon_day": horizon, "split": "test", "task": "rain_classification", "model": best_rain, "threshold": rain_threshold, "accuracy": accuracy_score(test["horizon_rain"], rain_test), "precision": precision_score(test["horizon_rain"], rain_test, zero_division=0), "recall": recall_score(test["horizon_rain"], rain_test, zero_division=0), "f1": f1_score(test["horizon_rain"], rain_test, zero_division=0)},
        ])
        horizons.append({"horizon_day": horizon, "temperature_model": horizon_info["temperature_models"][best_temp], "rain_model": horizon_info["rain_models"][best_rain], "temperature_model_name": best_temp, "rain_model_name": best_rain, "rain_threshold": rain_threshold})

    pd.DataFrame(metrics).to_csv(output_dir / "evaluation_metrics.csv", index=False)
    evaluation = {"contract_version": CONTRACT_VERSION, "stage": "evaluated", "created_at": datetime.now(timezone.utc).isoformat(), "feature_columns": feature_columns, "horizons": horizons, "metrics_path": "evaluation_metrics.csv", "metrics_path_json": "evaluation_metrics.json"}
    pd.DataFrame(metrics).to_json(output_dir / "evaluation_metrics.json", orient="records", indent=2)
    path = output_dir / "evaluation_manifest.json"
    path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return path
