"""MLflow Model Registry integration for the seven-day production bundle."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.production.provenance import git_commit, package_versions
from src.production.storage import load_feature_snapshot


def _mlflow_modules():
    try:
        import mlflow
        from mlflow import MlflowClient
        import mlflow.sklearn
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("MLflow is required for model registry operations") from exc
    return mlflow, MlflowClient


def _model_hyperparameters(model) -> dict[str, str]:
    """Return the fitted/search-selected parameters in MLflow-safe form."""
    if hasattr(model, "best_params_"):
        parameters = dict(model.best_params_)
        parameters.update({
            "search_cv": str(model.cv),
            "search_n_iter": str(getattr(model, "n_iter", "")),
            "search_scoring": str(getattr(model, "scoring", "")),
        })
    else:
        parameters = model.get_params(deep=True)
    return {f"hyperparameter_{key}": str(value) for key, value in parameters.items()}


def _log_evaluation_metrics(mlflow, metrics_path: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in metrics:
        prefix = f"h{int(row['horizon_day'])}_{row['task']}_{row['split']}"
        numeric_metrics = {
            key: float(value)
            for key, value in row.items()
            if key not in {"horizon_day", "task", "split", "model"}
            and isinstance(value, (int, float))
        }
        if numeric_metrics:
            mlflow.log_metrics({f"{prefix}_{key}": value for key, value in numeric_metrics.items()})


def _input_example(features_path: Path | None, feature_columns: list[str]):
    if not features_path or not Path(features_path).exists():
        return None
    frame = load_feature_snapshot(Path(features_path))
    return frame[feature_columns].head(1)


def _model_metrics(metrics: list[dict], horizon: int, task: str) -> dict[str, float]:
    result = {}
    for row in metrics:
        if int(row["horizon_day"]) != horizon or row["task"] != task:
            continue
        split = row["split"]
        for key, value in row.items():
            if key not in {"horizon_day", "task", "split", "model"} and isinstance(value, (int, float)):
                result[f"{split}_{key}"] = float(value)
    return result


def register_candidate_models(evaluation_manifest: Path, output_dir: Path, tracking_uri: str, experiment_name: str = "weather-7d-production", features_path: Path | None = None) -> Path:
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
    metrics = json.loads((evaluation_manifest.parent / manifest["metrics_path_json"]).read_text(encoding="utf-8"))
    comparison_path = evaluation_manifest.parent / manifest.get("candidate_comparison_path", "candidate_comparison.json")
    input_example = _input_example(features_path, manifest["feature_columns"])
    with mlflow.start_run(run_name="weather-7d-register-candidate") as run:
        parent_run_id = run.info.run_id
        mlflow.log_params({
            "contract_version": manifest["contract_version"],
            "stage": "candidate",
            "horizons": len(manifest["horizons"]),
            "feature_count": len(manifest.get("feature_columns", [])),
            "random_state": manifest.get("random_state", 42),
            "dataset_sha256": manifest.get("dataset_sha256", "unknown"),
            "feature_snapshot_sha256": manifest.get("feature_snapshot_sha256", "unknown"),
            "git_commit": git_commit(),
            **{f"search_{key}": value for key, value in manifest.get("search_config", {}).items()},
        })
        mlflow.set_tags({
            "contract_version": manifest["contract_version"],
            "git_commit": git_commit(),
            **{f"package_{key}": value for key, value in package_versions().items()},
        })
        mlflow.log_artifact(str(evaluation_manifest), artifact_path="evaluation")
        mlflow.log_artifact(str(output_dir / manifest["metrics_path"]), artifact_path="evaluation")
        if comparison_path.exists():
            mlflow.log_artifact(str(comparison_path), artifact_path="candidate_comparison")
        _log_evaluation_metrics(mlflow, output_dir / manifest["metrics_path_json"])
        for item in manifest["horizons"]:
            horizon = int(item["horizon_day"])
            for task, relative_path, model_name in (
                ("temperature", item["temperature_model"], f"weather-7d-temperature-h{horizon}"),
                ("rain", item["rain_model"], f"weather-7d-rain-h{horizon}"),
            ):
                model = joblib.load(output_dir / relative_path)
                artifact_path = f"model_h{horizon}_{task}"
                selected_algorithm = item[f"{task}_model_name"]
                with mlflow.start_run(run_name=f"h{horizon}-{task}-{selected_algorithm}", nested=True) as child_run:
                    child_run_id = child_run.info.run_id
                    mlflow.log_params({
                        "contract_version": manifest["contract_version"],
                        "horizon_day": horizon,
                        "task": task,
                        "algorithm": selected_algorithm,
                        "model_name": model_name,
                        "random_state": manifest.get("random_state", 42),
                        "dataset_sha256": manifest.get("dataset_sha256", "unknown"),
                        "feature_snapshot_sha256": manifest.get("feature_snapshot_sha256", "unknown"),
                        **_model_hyperparameters(model),
                    })
                    if task == "rain":
                        mlflow.log_param("decision_threshold", item["rain_threshold"])
                    child_metrics = _model_metrics(metrics, horizon, "temperature_regression" if task == "temperature" else "rain_classification")
                    mlflow.log_metrics(child_metrics)
                    if input_example is not None:
                        prediction = model.predict(input_example)
                        signature = mlflow.models.infer_signature(input_example, prediction)
                        start = time.perf_counter()
                        model.predict(input_example)
                        latency_ms = (time.perf_counter() - start) * 1000
                        mlflow.log_metric("prediction_latency_ms", latency_ms)
                        mlflow.sklearn.log_model(model, artifact_path=artifact_path, signature=signature, input_example=input_example)
                    else:
                        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
                    mlflow.log_metric("model_size_bytes", float((output_dir / relative_path).stat().st_size))
                    model_uri = f"runs:/{child_run_id}/{artifact_path}"
                    registered = mlflow.register_model(model_uri, model_name, await_registration_for=0)
                version = str(registered.version)
                client.set_registered_model_alias(model_name, "candidate", version)
                client.set_model_version_tag(model_name, version, "contract_version", manifest["contract_version"])
                client.set_model_version_tag(model_name, version, "horizon_day", str(horizon))
                client.set_model_version_tag(model_name, version, "task", task)
                client.set_model_version_tag(model_name, version, "algorithm", selected_algorithm)
                records.append({"horizon_day": horizon, "task": task, "name": model_name, "version": version, "run_id": child_run_id, "parent_run_id": parent_run_id, "model_uri": model_uri})

    path = output_dir / "registry_manifest.json"
    path.write_text(json.dumps({
        "contract_version": manifest["contract_version"],
        "stage": "candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "parent_run_id": parent_run_id,
        "source_evaluation_manifest": str(evaluation_manifest),
        "models": records,
    }, indent=2), encoding="utf-8")
    return path


def promote_registry_aliases(registry_manifest: Path, tracking_uri: str, promotion_thresholds: dict[str, float] | None = None) -> dict:
    """Move candidate to champion and preserve the previous champion alias."""
    _, MlflowClient = _mlflow_modules()
    payload = json.loads(Path(registry_manifest).read_text(encoding="utf-8"))
    client = MlflowClient(tracking_uri=tracking_uri)
    parent_run_id = payload.get("parent_run_id")
    if parent_run_id and promotion_thresholds:
        client.log_param(parent_run_id, "promotion_max_test_rmse", promotion_thresholds["max_test_rmse"])
        client.log_param(parent_run_id, "promotion_min_test_f1", promotion_thresholds["min_test_f1"])
        client.set_tag(parent_run_id, "promotion_gate", "passed")
    promoted = []
    for item in payload["models"]:
        name = item["name"]
        try:
            current = client.get_model_version_by_alias(name, "champion")
            client.set_registered_model_alias(name, "previous", str(current.version))
        except Exception:
            pass
        client.set_registered_model_alias(name, "champion", str(item["version"]))
        # A promoted version must not retain the candidate alias. The
        # candidate alias represents the version waiting for promotion;
        # champion is the only production alias.
        client.delete_registered_model_alias(name, "candidate")
        promoted.append({**item, "alias": "champion"})
    return {"stage": "production", "models": promoted}
