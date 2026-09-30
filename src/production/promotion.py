"""Explicit candidate-to-champion promotion gate."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.production.registry import promote_registry_aliases


def promote(evaluation_manifest: Path, output_dir: Path, max_rmse: float = 4.0, min_f1: float = 0.55, registry_manifest: Path | None = None, tracking_uri: str | None = None) -> Path:
    evaluation_manifest = Path(evaluation_manifest)
    output_dir = Path(output_dir)
    manifest = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    if manifest.get("stage") != "evaluated":
        raise ValueError("promotion requires an evaluated manifest")
    metrics = json.loads((evaluation_manifest.parent / manifest["metrics_path_json"]).read_text(encoding="utf-8"))
    failures = [row for row in metrics if row["split"] == "test" and ((row.get("rmse") is not None and row["rmse"] > max_rmse) or (row.get("f1") is not None and row["f1"] < min_f1))]
    if failures:
        raise ValueError(f"promotion gate rejected candidate: {len(failures)} metric rows failed")
    production = dict(manifest)
    production["stage"] = "production"
    production["promotion"] = {"max_test_rmse": max_rmse, "min_test_f1": min_f1, "gate": "passed"}
    if registry_manifest and tracking_uri:
        production["registry"] = promote_registry_aliases(
            registry_manifest,
            tracking_uri,
            {"max_test_rmse": max_rmse, "min_test_f1": min_f1},
        )
    path = output_dir / "production_manifest.json"
    path.write_text(json.dumps(production, indent=2), encoding="utf-8")
    shutil.copy2(path, output_dir / "champion_manifest.json")
    return path
