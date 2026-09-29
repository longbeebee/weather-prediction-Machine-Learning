"""Airflow callbacks that route task status notifications through Alertmanager."""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone


def _send_airflow_alert(context: dict, *, event: str, severity: str, ends_in_minutes: int) -> None:
    task_instance = context.get("task_instance")
    dag_id = task_instance.dag_id if task_instance else "unknown"
    task_id = task_instance.task_id if task_instance else "unknown"
    run_id = context.get("run_id", "unknown")
    exception = str(context.get("exception", "unknown error"))
    now = datetime.now(timezone.utc)
    alert_name = "AirflowTaskSucceeded" if event == "success" else "AirflowTaskFailed"
    summary = f"Airflow task succeeded: {dag_id}.{task_id}" if event == "success" else f"Airflow task failed: {dag_id}.{task_id}"
    description = f"run_id={run_id}"
    if event != "success":
        description += f"; exception={exception[:1000]}"
    alert = [{
        "labels": {
            "alertname": alert_name,
            "severity": severity,
            "source": "airflow",
            "event": event,
            "dag_id": dag_id,
            "task_id": task_id,
            "run_id": run_id,
        },
        "annotations": {
            "summary": summary,
            "description": description,
        },
        "startsAt": now.isoformat(),
        "endsAt": (now + timedelta(minutes=ends_in_minutes)).isoformat(),
    }]
    payload = json.dumps(alert).encode("utf-8")
    endpoint = os.getenv("ALERTMANAGER_URL", "http://alertmanager:9093") + "/api/v2/alerts"
    request = urllib.request.Request(endpoint, data=payload, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status >= 300:
                raise RuntimeError(f"Alertmanager returned HTTP {response.status}")
    except Exception as exc:
        # Never hide the original Airflow task failure when notification fails.
        print(f"Airflow {event} notification failed: {exc}")


def airflow_task_failure_callback(context: dict) -> None:
    """Notify Telegram when an Airflow task fails."""
    _send_airflow_alert(context, event="failed", severity="critical", ends_in_minutes=5)


def airflow_task_success_callback(context: dict) -> None:
    """Notify Telegram when an Airflow task succeeds."""
    _send_airflow_alert(context, event="success", severity="info", ends_in_minutes=1)
