from __future__ import annotations

import argparse
from pathlib import Path

from src.production.monitoring import apply_drift_result, run_evidently_drift_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Evidently feature drift monitoring")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--html-report", type=Path, required=True)
    parser.add_argument("--json-report", type=Path, required=True)
    parser.add_argument("--current-features-path", type=Path, default=None)
    parser.add_argument("--performance-report", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--pushgateway-url", default=None)
    parser.add_argument("--release-id", default=None)
    args = parser.parse_args()
    summary = run_evidently_drift_report(args.data_path, args.html_report, args.json_report, args.current_features_path, release_id=args.release_id)
    apply_drift_result(args.performance_report, summary, args.decision, args.pushgateway_url)
    print(summary)


if __name__ == "__main__":
    main()
