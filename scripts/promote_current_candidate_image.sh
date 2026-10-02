#!/usr/bin/env bash
set -euo pipefail

candidate_container="$(docker compose ps -q weather-api-canary)"
if [ -z "$candidate_container" ]; then
  echo "weather-api-canary container is not running" >&2
  exit 1
fi

candidate_image="$(docker inspect -f '{{.Config.Image}}' "$candidate_container")"
if [ -z "$candidate_image" ] || [ "$candidate_image" = "weather-api:local" ]; then
  echo "candidate is not using an ECR image: ${candidate_image:-<empty>}" >&2
  exit 1
fi

echo "Recreating champion with image: $candidate_image"
WEATHER_API_IMAGE="$candidate_image" \
  docker compose up -d --force-recreate weather-api

for _ in $(seq 1 30); do
  if curl -fsS http://localhost:8000/ready >/dev/null; then
    echo "Champion is ready with image: $candidate_image"
    exit 0
  fi
  sleep 2
done

docker compose logs --tail=200 weather-api
exit 1
