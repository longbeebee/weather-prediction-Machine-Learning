"""Airflow monitoring DAG for the published seven-day contract."""

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

from callbacks import airflow_task_failure_callback, airflow_task_success_callback


with DAG(
    dag_id="weather_7d_monitoring",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["weather", "monitoring"],
    default_args={
        "on_failure_callback": airflow_task_failure_callback,
        "on_success_callback": airflow_task_success_callback,
    },
) as dag:
    check_manifest = BashOperator(
        task_id="check_production_manifest",
        bash_command="python -c \"import json; p='/opt/airflow/models/seven_day_production/production_manifest.json'; m=json.load(open(p)); assert m['stage']=='production' and len(m['horizons'])==7\"",
    )
    collect_actuals = BashOperator(
        task_id="collect_actuals_from_open_meteo",
        bash_command="python -m src.production.cli_actuals --predictions-jsonl /opt/airflow/project/monitoring/predictions/api_predictions.jsonl --output-csv /opt/airflow/run/actuals.csv --metadata-path /opt/airflow/run/actuals_metadata.json --latitude \"$WEATHER_LATITUDE\" --longitude \"$WEATHER_LONGITUDE\" --timezone \"$WEATHER_TIMEZONE\"",
    )
    materialize_datalake = BashOperator(
        task_id="materialize_datalake",
        bash_command="python -m src.production.cli_datalake --predictions-jsonl /opt/airflow/project/monitoring/predictions/api_predictions.jsonl --actuals-csv /opt/airflow/run/actuals.csv --output-dir /opt/airflow/run/datalake --s3-bucket \"$S3_STORAGE_BUCKET\" --s3-prefix \"weather-7d/bronze/event_date={{ ds }}\" --region \"$AWS_DEFAULT_REGION\"",
    )
    calculate_performance = BashOperator(
        task_id="calculate_performance",
        bash_command="python -m src.production.cli_monitor --dataset /opt/airflow/run/datalake/monitoring_dataset.parquet --report /opt/airflow/monitoring/performance_report.json --decision /opt/airflow/monitoring/retraining_decision.json --pushgateway-url http://pushgateway:9091",
    )
    run_drift_checks = BashOperator(
        task_id="run_drift_checks",
        bash_command="python -m src.production.cli_drift --data-path /opt/airflow/project/data/raw/open_meteo_hanoi.csv --current-features-path /opt/airflow/run/datalake/predictions.parquet --html-report /opt/airflow/project/monitoring/evidently/latest_drift_report.html --json-report /opt/airflow/project/monitoring/evidently/latest_drift_report.json --performance-report /opt/airflow/monitoring/performance_report.json --decision /opt/airflow/monitoring/retraining_decision.json --pushgateway-url http://pushgateway:9091",
    )
    archive_monitoring = BashOperator(
        task_id="archive_monitoring_artifacts",
        bash_command="python -m src.production.cli_archive --path /opt/airflow/run/datalake --path /opt/airflow/run/actuals_metadata.json --path /opt/airflow/monitoring/performance_report.json --path /opt/airflow/monitoring/retraining_decision.json --path /opt/airflow/project/monitoring/evidently --s3-bucket \"$S3_STORAGE_BUCKET\" --s3-prefix \"weather-7d/runs/{{ ts_nodash }}/monitoring\" --region \"$AWS_DEFAULT_REGION\"",
    )
    check_manifest >> collect_actuals >> materialize_datalake >> calculate_performance >> run_drift_checks >> archive_monitoring
