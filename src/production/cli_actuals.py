from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.production.actuals import collect_actuals


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect historical actual weather for production predictions")
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--latitude", type=float, default=_env_float("WEATHER_LATITUDE", 21.0285))
    parser.add_argument("--longitude", type=float, default=_env_float("WEATHER_LONGITUDE", 105.8542))
    parser.add_argument("--timezone", default=os.getenv("WEATHER_TIMEZONE") or "Asia/Bangkok")
    args = parser.parse_args()
    print(collect_actuals(args.predictions_jsonl, args.output_csv, args.metadata_path, args.latitude, args.longitude, args.timezone))


if __name__ == "__main__":
    main()
