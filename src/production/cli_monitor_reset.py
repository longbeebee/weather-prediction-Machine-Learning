from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.production.monitoring import reset_monitoring_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize monitoring metrics for the active production release")
    parser.add_argument("--production-manifest", type=Path, default=Path("models/seven_day_production/production_manifest.json"))
    parser.add_argument("--pushgateway-url", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.production_manifest.read_text(encoding="utf-8"))
    release_id = manifest.get("release_id")
    if not release_id:
        raise ValueError("production manifest is missing release_id")
    reset_monitoring_metrics(args.pushgateway_url, release_id=str(release_id))
    print(f"monitoring metrics reset for release_id={release_id}")


if __name__ == "__main__":
    main()
