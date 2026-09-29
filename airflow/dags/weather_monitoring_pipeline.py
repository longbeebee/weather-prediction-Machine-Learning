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
    calculate_performance = BashOperator(
        task_id="calculate_performance",
        bash_command="python -m src.production.cli_monitor --predictions /opt/airflow/monitoring/predictions.csv --actuals /opt/airflow/monitoring/actuals.csv --report /opt/airflow/monitoring/performance_report.json --decision /opt/airflow/monitoring/retraining_decision.json --pushgateway-url http://pushgateway:9091",
    )
    check_manifest >> calculate_performance
