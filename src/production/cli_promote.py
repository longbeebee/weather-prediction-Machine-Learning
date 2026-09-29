from __future__ import annotations

import argparse
from pathlib import Path

from src.production.promotion import promote


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote an evaluated 7-day model bundle")
    parser.add_argument("--evaluation-manifest", type=Path, default=Path("models/seven_day_production/evaluation_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/seven_day_production"))
    parser.add_argument("--max-rmse", type=float, default=4.0)
    parser.add_argument("--min-f1", type=float, default=0.55)
    parser.add_argument("--registry-manifest", type=Path, default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    args = parser.parse_args()
    print(promote(args.evaluation_manifest, args.output_dir, args.max_rmse, args.min_f1, args.registry_manifest, args.mlflow_tracking_uri))


if __name__ == "__main__":
    main()
