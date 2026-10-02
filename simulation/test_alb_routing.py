"""Measure ALB weighted routing between champion and canary APIs."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from generate_prediction_logs import build_request


def call_api(url: str, payload: dict, timeout: int) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body)
        except json.JSONDecodeError:
            detail = {"error": body}
        return exc.code, detail
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return 0, {"error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure champion/canary traffic split through an ALB"
    )
    parser.add_argument("--url", default=os.getenv("ALB_PREDICTION_API_URL"), required=not os.getenv("ALB_PREDICTION_API_URL"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--delay-seconds", type=float, default=0.02)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("simulation/alb_routing_responses.jsonl"))
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be greater than zero")
    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds cannot be negative")

    now = datetime.now(ZoneInfo("Asia/Bangkok")).replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=None)
    first_latest = now - timedelta(days=args.count + 8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tracks = Counter()
    statuses = Counter()

    with args.output.open("w", encoding="utf-8") as output:
        for index in range(args.count):
            status, response = call_api(
                args.url,
                build_request(first_latest + timedelta(days=index), index),
                args.timeout,
            )
            track = response.get("model_track", "unknown")
            tracks[track] += 1
            statuses[status] += 1
            output.write(json.dumps({
                "request_index": index + 1,
                "http_status": status,
                "model_track": track,
                "request_id": response.get("request_id"),
                "response": response,
            }, default=str) + "\n")
            if (index + 1) % 100 == 0 or index + 1 == args.count:
                print(f"completed={index + 1}/{args.count}")
            if args.delay_seconds:
                time.sleep(args.delay_seconds)

    print("\nRouting summary:")
    for track, count in sorted(tracks.items()):
        print(f"  {track}: {count} ({count / args.count * 100:.2f}%)")
    print(f"HTTP statuses: {dict(statuses)}")
    print(f"response_log={args.output}")


if __name__ == "__main__":
    main()
