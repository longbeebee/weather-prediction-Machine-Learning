#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?AWS_REGION is required}"
: "${ECR_REGISTRY:?ECR_REGISTRY is required}"
: "${ECR_REPOSITORY:?ECR_REPOSITORY is required}"
: "${IMAGE_TAG:?IMAGE_TAG is required}"

image="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# Preserve the active canary release when recreating API containers. Model
# routing and MLflow aliases remain explicit operations in the canary pipeline.
canary_manifest="$(
  container_id="$(docker compose ps -q weather-api-canary 2>/dev/null || true)"
  if [ -n "$container_id" ]; then
    docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$container_id" \
      | sed -n 's/^CANARY_MODEL_MANIFEST=//p' | head -n 1
  fi
)"
canary_manifest="${canary_manifest:-/app/models/seven_day_production/candidate_serving_manifest.json}"

WEATHER_API_IMAGE="$image" \
CANARY_MODEL_MANIFEST="$canary_manifest" \
  docker compose pull weather-api weather-api-canary

# Airflow source is bind-mounted; dependency changes require a fresh image.
# Stateful services and databases are not rebuilt by this deployment.
docker compose build airflow-init

WEATHER_API_IMAGE="$image" \
CANARY_MODEL_MANIFEST="$canary_manifest" \
  docker compose up -d --force-recreate \
    weather-api weather-api-canary airflow-webserver airflow-scheduler \
    prometheus grafana alertmanager

wait_for_ready() {
  local url="$1"
  local service="$2"
  for _ in $(seq 1 30); do
    if curl -fsS "$url" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  docker compose logs --tail=200 "$service"
  return 1
}

wait_for_ready "http://localhost:8000/ready" weather-api
wait_for_ready "http://localhost:8001/ready" weather-api-canary

echo "Application deployment completed: image=$image canary_manifest=$canary_manifest"
