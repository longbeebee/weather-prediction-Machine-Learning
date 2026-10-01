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
    max_active_runs=1,
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
    build_features = BashOperator(
        task_id="build_and_store_features",
        bash_command="python -m src.production.cli_features --data-path /opt/airflow/run/raw.csv --output-path /opt/airflow/run/features.parquet --metadata-path /opt/airflow/run/feature_metadata.json --s3-bucket \"$S3_FEATURE_BUCKET\" --s3-prefix \"$S3_FEATURE_PREFIX\" --region \"$AWS_DEFAULT_REGION\"",
    )
    train = BashOperator(
        task_id="train_candidates",
        bash_command="python -m src.production.cli_train --output-dir /opt/airflow/models/seven_day_production --data-path /opt/airflow/run/raw.csv --features-path /opt/airflow/run/features.parquet --mlflow-tracking-uri $MLFLOW_TRACKING_URI",
    )
    evaluate = BashOperator(
        task_id="evaluate_candidates",
        bash_command="python -m src.production.cli_evaluate --candidate-manifest /opt/airflow/models/seven_day_production/candidate_manifest.json --output-dir /opt/airflow/models/seven_day_production --data-path /opt/airflow/run/raw.csv --features-path /opt/airflow/run/features.parquet --mlflow-tracking-uri $MLFLOW_TRACKING_URI",
    )
    register = BashOperator(
        task_id="register_candidate",
        bash_command="python -m src.production.cli_register --evaluation-manifest /opt/airflow/models/seven_day_production/evaluation_manifest.json --output-dir /opt/airflow/models/seven_day_production --features-path /opt/airflow/run/features.parquet --mlflow-tracking-uri $MLFLOW_TRACKING_URI",
    )
    archive_candidate = BashOperator(
        task_id="archive_candidate_artifacts",
        bash_command="python -m src.production.cli_archive --path /opt/airflow/run/raw.csv --path /opt/airflow/run/validation.json --path /opt/airflow/run/features.parquet --path /opt/airflow/run/feature_metadata.json --path /opt/airflow/models/seven_day_production/candidates --path /opt/airflow/models/seven_day_production/candidate_manifest.json --path /opt/airflow/models/seven_day_production/evaluation_metrics.csv --path /opt/airflow/models/seven_day_production/evaluation_metrics.json --path /opt/airflow/models/seven_day_production/evaluation_manifest.json --path /opt/airflow/models/seven_day_production/candidate_comparison.json --path /opt/airflow/models/seven_day_production/registry_manifest.json --s3-bucket \"$S3_STORAGE_BUCKET\" --s3-prefix \"weather-7d/runs/{{ ts_nodash }}/training/candidate\" --region \"$AWS_DEFAULT_REGION\"",
    )
    # Promotion is intentionally outside the scheduled training DAG.  The
    # evaluated candidate must first be deployed behind the ALB and observed
    # as canary traffic.  Jenkins' parameterized promotion job performs the
    # final candidate -> champion transition after that review.
    ingest >> validate >> build_features >> train >> evaluate >> register >> archive_candidate
