"""MLflow Model Registry integration for the seven-day production bundle."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib


def _mlflow_modules():
    try:
        import mlflow
        from mlflow import MlflowClient
        import mlflow.sklearn
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("MLflow is required for model registry operations") from exc
    return mlflow, MlflowClient


def register_candidate_models(evaluation_manifest: Path, output_dir: Path, tracking_uri: str, experiment_name: str = "weather-7d-production") -> Path:
    """Register evaluated horizon models without changing the champion."""
    mlflow, MlflowClient = _mlflow_modules()
    evaluation_manifest = Path(evaluation_manifest)
    output_dir = Path(output_dir)
    manifest = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    if manifest.get("stage") != "evaluated":
        raise ValueError("registry registration requires an evaluated manifest")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    records: list[dict] = []
    with mlflow.start_run(run_name="weather-7d-register-candidate") as run:
        mlflow.log_params({"contract_version": manifest["contract_version"], "stage": "candidate", "horizons": len(manifest["horizons"])})
        mlflow.log_artifact(str(evaluation_manifest), artifact_path="evaluation")
        mlflow.log_artifact(str(output_dir / manifest["metrics_path"]), artifact_path="evaluation")
        for item in manifest["horizons"]:
            horizon = int(item["horizon_day"])
            for task, relative_path, model_name in (
                ("temperature", item["temperature_model"], f"weather-7d-temperature-h{horizon}"),
                ("rain", item["rain_model"], f"weather-7d-rain-h{horizon}"),
            ):
                model = joblib.load(output_dir / relative_path)
                artifact_path = f"model_h{horizon}_{task}"
                mlflow.sklearn.log_model(model, artifact_path=artifact_path)
                model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
                registered = mlflow.register_model(model_uri, model_name, await_registration_for=0)
                version = str(registered.version)
                client.set_registered_model_alias(model_name, "candidate", version)
                records.append({"horizon_day": horizon, "task": task, "name": model_name, "version": version, "run_id": run.info.run_id, "model_uri": model_uri})

    path = output_dir / "registry_manifest.json"
    path.write_text(json.dumps({
        "contract_version": manifest["contract_version"],
        "stage": "candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "source_evaluation_manifest": str(evaluation_manifest),
        "models": records,
    }, indent=2), encoding="utf-8")
    return path


def promote_registry_aliases(registry_manifest: Path, tracking_uri: str) -> dict:
    """Move candidate to champion and preserve the previous champion alias."""
    _, MlflowClient = _mlflow_modules()
    payload = json.loads(Path(registry_manifest).read_text(encoding="utf-8"))
    client = MlflowClient(tracking_uri=tracking_uri)
    promoted = []
    for item in payload["models"]:
        name = item["name"]
        try:
            current = client.get_model_version_by_alias(name, "champion")
            client.set_registered_model_alias(name, "previous", str(current.version))
        except Exception:
            pass
        client.set_registered_model_alias(name, "champion", str(item["version"]))
        promoted.append({**item, "alias": "champion"})
    return {"stage": "production", "models": promoted}
