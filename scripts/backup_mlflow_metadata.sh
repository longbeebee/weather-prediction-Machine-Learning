#!/usr/bin/env bash
set -euo pipefail

: "${S3_STORAGE_BUCKET:?S3_STORAGE_BUCKET is required}"
backup_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_key="s3://${S3_STORAGE_BUCKET}/weather-7d/backups/mlflow/metadata-${backup_stamp}.sql.gz"

docker compose exec -T \
  -e PGPASSWORD="${MLFLOW_DB_PASSWORD:-mlflow-db-demo}" \
  postgres-airflow pg_dump -U mlflow -d mlflow --no-owner --no-privileges \
  | gzip \
  | aws s3 cp - "${backup_key}"

echo "MLflow metadata backup uploaded to ${backup_key}"
