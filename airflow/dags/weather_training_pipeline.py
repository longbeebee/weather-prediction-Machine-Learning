"""Airflow training DAG with explicit quality and promotion boundaries."""

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

from callbacks import airflow_task_failure_callback, airflow_task_success_callback


with DAG(
    dag_id="weather_7d_training",
    start_date=datetime(2025, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["weather", "mlops"],
    default_args={
        "on_failure_callback": airflow_task_failure_callback,
        "on_success_callback": airflow_task_success_callback,
    },
) as dag:
    ingest = BashOperator(
        task_id="ingest",
        bash_command="mkdir -p /opt/airflow/run && cp /opt/airflow/project/data/raw/open_meteo_hanoi.csv /opt/airflow/run/raw.csv",
    )
    validate = BashOperator(
        task_id="validate_data",
        bash_command="python -m src.production.cli_validate --data-path /opt/airflow/run/raw.csv --report-path /opt/airflow/run/validation.json",
    )
    train = BashOperator(
        task_id="train_candidates",
        bash_command="python -m src.production.cli_train --output-dir /opt/airflow/models/seven_day_production --data-path /opt/airflow/run/raw.csv",
    )
    evaluate = BashOperator(
        task_id="evaluate_candidates",
        bash_command="python -m src.production.cli_evaluate --candidate-manifest /opt/airflow/models/seven_day_production/candidate_manifest.json --output-dir /opt/airflow/models/seven_day_production --data-path /opt/airflow/run/raw.csv",
    )
    promote = BashOperator(
        task_id="promote_candidate",
        bash_command="python -m src.production.cli_promote --evaluation-manifest /opt/airflow/models/seven_day_production/evaluation_manifest.json --output-dir /opt/airflow/models/seven_day_production",
    )
    ingest >> validate >> train >> evaluate >> promote
