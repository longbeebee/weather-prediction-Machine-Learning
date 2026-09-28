from __future__ import annotations

import argparse
from pathlib import Path

from src.production.training import train_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 7-day candidate models only")
    parser.add_argument("--output-dir", type=Path, default=Path("models/seven_day_production"))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    args = parser.parse_args()
    manifest = train_candidates(args.output_dir, args.data_path)
    if args.mlflow_tracking_uri:
        from src.production.tracking import log_json_artifact
        run_id = log_json_artifact(manifest, args.mlflow_tracking_uri, "weather-7d-training", {"stage": "candidate"})
        print(f"mlflow_run_id={run_id}")
    print(manifest)


if __name__ == "__main__":
    main()
