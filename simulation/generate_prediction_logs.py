"""Generate synthetic historical requests against the production prediction API.

The requests are intentionally dated in the past so their seven forecast
timestamps can be resolved by the actuals collector during a demo run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


def build_request(latest_time: datetime, offset: int) -> dict:
    observations = []
    for index in range(8):
        timestamp = latest_time - timedelta(days=7 - index)
        observations.append({
            "time": timestamp.isoformat(),
            "temperature": 22.0 + ((index + offset) % 7) * 0.8,
            "humidity": 62.0 + ((index * 3 + offset) % 18),
            "precipitation": 0.4 if (index + offset) % 9 == 0 else 0.0,
            "cloud_cover": 45.0 + ((index * 5 + offset) % 45),
            "pressure_msl": 1010.0 + ((index + offset) % 6),
            "wind_speed_10m": 4.0 + ((index + offset) % 5),
            "shortwave_radiation": 350.0 + ((index * 20 + offset) % 250),
        })
    return {"observations": observations}


def call_api(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return {"status": response.status, "response": json.loads(body)}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned HTTP {exc.code}: {body}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate historical prediction logs through the production API")
    parser.add_argument("--url", default=os.getenv("PREDICTION_API_URL", "http://localhost:8000/api/v1/predict"))
    parser.add_argument("--count", type=int, default=35, help="Number of prediction requests; Evidently needs at least 30")
    parser.add_argument("--days-behind", type=int, default=8, help="Keep the latest forecast this many days in the past")
    parser.add_argument("--delay-seconds", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("simulation/prediction_responses.jsonl"))
    args = parser.parse_args()
    if args.count < 30:
        raise ValueError("--count must be at least 30 for the default Evidently window")

    now = datetime.now(ZoneInfo("Asia/Bangkok")).replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=None)
    first_latest = now - timedelta(days=args.days_behind + args.count - 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    with args.output.open("w", encoding="utf-8") as output:
        for offset in range(args.count):
            latest_time = first_latest + timedelta(days=offset)
            result = call_api(args.url, build_request(latest_time, offset), args.timeout)
            response = result["response"]
            output.write(json.dumps(response, default=str) + "\n")
            output.flush()
            successes += 1
            print(f"[{successes}/{args.count}] request_id={response.get('request_id')} latest_observation={latest_time.isoformat()}")
            if args.delay_seconds:
                time.sleep(args.delay_seconds)
    print(f"completed={successes} response_log={args.output}")


if __name__ == "__main__":
    main()
