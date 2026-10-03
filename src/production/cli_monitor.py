from __future__ import annotations

import argparse
from pathlib import Path

from src.production.monitoring import decide_retraining, monitor_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate production performance and retraining decision")
    parser.add_argument("--dataset", type=Path, required=True, help="Canonical Parquet dataset containing predictions and actuals")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--pushgateway-url", default=None)
    parser.add_argument("--release-id", default=None)
    args = parser.parse_args()
    report = monitor_predictions(args.dataset, args.report, release_id=args.release_id)
    print(decide_retraining(report, args.decision))
    if args.pushgateway_url:
        from src.production.monitoring import push_monitoring_metrics
        push_monitoring_metrics(report, args.pushgateway_url)


if __name__ == "__main__":
    main()
