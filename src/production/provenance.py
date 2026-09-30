"""Reproducibility metadata shared by training, evaluation, and MLflow."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    for directory in (Path.cwd(), Path("/opt/airflow/project")):
        try:
            return subprocess.check_output(
                ["git", "-C", str(directory), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
    return "unknown"


def package_versions(names: tuple[str, ...] = ("mlflow", "scikit-learn", "xgboost", "pandas", "numpy", "evidently", "duckdb")) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def split_configuration(frame) -> dict:
    years = set(frame["time"].dt.year)
    if {2020, 2021, 2022, 2023, 2024, 2025}.issubset(years):
        return {
            "strategy": "calendar_year",
            "train_rule": "year <= 2023",
            "validation_rule": "year == 2024",
            "test_rule": "year == 2025",
        }
    return {
        "strategy": "chronological_ratio",
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
    }
