from __future__ import annotations

import argparse
from pathlib import Path

from src.production.registry import register_candidate_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Register evaluated seven-day models as MLflow candidates")
    parser.add_argument("--evaluation-manifest", type=Path, default=Path("models/seven_day_production/evaluation_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/seven_day_production"))
    parser.add_argument("--mlflow-tracking-uri", required=True)
    parser.add_argument("--features-path", type=Path, default=None)
    args = parser.parse_args()
    print(register_candidate_models(args.evaluation_manifest, args.output_dir, args.mlflow_tracking_uri, features_path=args.features_path))


if __name__ == "__main__":
    main()
