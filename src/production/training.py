"""Candidate training stage for the versioned 7-day production contract.

This module intentionally does not evaluate validation/test data or select a
production model. It only fits candidates on the training split and persists
them for the next, explicit evaluation stage.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from src.production.storage import load_feature_snapshot

from src.config import FALLBACK_RAW_DATA_PATH, RAW_DATA_PATH
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.production.features import build_base_features, build_horizon_targets, merge_horizon_targets, select_feature_columns
from src.seven_day_candidates import train_rain_candidates, train_temperature_candidates
from src.split import chronological_split


CONTRACT_VERSION = "weather-7d-v1"


def train_candidates(output_dir: Path, data_path: Path | None = None, features_path: Path | None = None) -> Path:
    """Fit all candidate models and return the candidate manifest path."""
    output_dir = Path(output_dir)
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_path or (RAW_DATA_PATH if RAW_DATA_PATH.exists() else FALLBACK_RAW_DATA_PATH)

    raw = DataLoader(csv_path).load()
    noon = Preprocessor().preprocess(raw, save=False)
    base_features = load_feature_snapshot(features_path) if features_path else build_base_features(noon)
    feature_columns = select_feature_columns(base_features)
    train_base, _, _ = chronological_split(base_features)

    horizons = []
    for horizon in range(1, 8):
        frame = merge_horizon_targets(base_features, build_horizon_targets(noon, horizon))
        train_frame, _, _ = chronological_split(frame)
        X_train = train_frame[feature_columns]
        temperature_models, _ = train_temperature_candidates(horizon, X_train, train_frame["horizon_temperature"])
        rain_models, _ = train_rain_candidates(horizon, X_train, train_frame["horizon_rain"].astype(int))

        horizon_dir = candidate_dir / f"horizon_{horizon}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        temperature_paths = {}
        for name, model in temperature_models.items():
            path = horizon_dir / f"temperature_{name}.joblib"
            joblib.dump(model, path)
            temperature_paths[name] = str(path.relative_to(output_dir))
        rain_paths = {}
        for name, model in rain_models.items():
            path = horizon_dir / f"rain_{name}.joblib"
            joblib.dump(model, path)
            rain_paths[name] = str(path.relative_to(output_dir))
        horizons.append({"horizon_day": horizon, "temperature_models": temperature_paths, "rain_models": rain_paths})

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "stage": "candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(csv_path),
        "data_rows": int(len(raw)),
        "noon_rows": int(len(noon)),
        "train_rows": int(len(train_base)),
        "feature_columns": feature_columns,
        "horizons": horizons,
    }
    path = output_dir / "candidate_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
