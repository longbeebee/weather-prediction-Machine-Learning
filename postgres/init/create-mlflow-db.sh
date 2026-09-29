#!/bin/sh
set -eu

export PGPASSWORD="${POSTGRES_PASSWORD}"
psql_base="psql -h "${POSTGRES_HOST}" -U airflow -d postgres -v ON_ERROR_STOP=1"

if ! ${psql_base} -tAc "SELECT 1 FROM pg_roles WHERE rolname='mlflow'" | grep -q 1; then
  ${psql_base} --set=mlflow_password="${MLFLOW_DB_PASSWORD}" \
    -c "CREATE ROLE mlflow LOGIN PASSWORD :'mlflow_password'"
fi

if ! ${psql_base} -tAc "SELECT 1 FROM pg_database WHERE datname='mlflow'" | grep -q 1; then
  createdb -h "${POSTGRES_HOST}" -U airflow -O mlflow mlflow
fi

echo "MLflow database and role are ready"
