"""Optional MLflow integration for candidate/evaluation artifacts.

The local pipeline remains usable without MLflow. In deployment, pass an
MLflow tracking URI to the CLI commands to record the same manifests and
metrics used by the API promotion boundary.
"""

from __future__ import annotations

import json
from pathlib import Path


def log_json_artifact(path: Path, tracking_uri: str, run_name: str, parameters: dict | None = None) -> str:
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - exercised in deployment image
        raise RuntimeError("MLflow tracking requested but mlflow is not installed") from exc
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_name=run_name) as run:
        if parameters:
            mlflow.log_params({key: str(value) for key, value in parameters.items()})
        mlflow.log_artifact(str(path))
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            mlflow.log_dict(payload, f"metadata/{path.name}")
        return run.info.run_id


def log_evaluation_artifacts(output_dir: Path, tracking_uri: str) -> str:
    """Log evaluation evidence without changing the production manifest."""
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MLflow tracking requested but mlflow is not installed") from exc
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_name="weather-7d-evaluation") as run:
        metrics_path = output_dir / "evaluation_metrics.csv"
        manifest_path = output_dir / "evaluation_manifest.json"
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        mlflow.log_artifact(str(manifest_path), artifact_path="promotion")
        return run.info.run_id
