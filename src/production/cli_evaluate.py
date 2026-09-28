from __future__ import annotations

import argparse
from pathlib import Path

from src.production.evaluation import evaluate_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 7-day candidates and publish production manifest")
    parser.add_argument("--candidate-manifest", type=Path, default=Path("models/seven_day_production/candidate_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/seven_day_production"))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    args = parser.parse_args()
    manifest = evaluate_candidates(args.candidate_manifest, args.output_dir, args.data_path)
    if args.mlflow_tracking_uri:
        from src.production.tracking import log_evaluation_artifacts
        run_id = log_evaluation_artifacts(args.output_dir, args.mlflow_tracking_uri)
        print(f"mlflow_run_id={run_id}")
    print(manifest)


if __name__ == "__main__":
    main()
