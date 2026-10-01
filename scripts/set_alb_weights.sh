#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?AWS_REGION is required}"
: "${ALB_LISTENER_ARN:?ALB_LISTENER_ARN is required}"
: "${CHAMPION_TG_ARN:?CHAMPION_TG_ARN is required}"
: "${CANDIDATE_TG_ARN:?CANDIDATE_TG_ARN is required}"

champion_weight="${1:?usage: set_alb_weights.sh <champion-weight> <candidate-weight>}"
candidate_weight="${2:?usage: set_alb_weights.sh <champion-weight> <candidate-weight>}"

case "$champion_weight" in ''|*[!0-9]*) echo "invalid champion weight" >&2; exit 2 ;; esac
case "$candidate_weight" in ''|*[!0-9]*) echo "invalid candidate weight" >&2; exit 2 ;; esac

if (( champion_weight > 999 || candidate_weight > 999 || champion_weight + candidate_weight == 0 )); then
  echo "weights must be between 0 and 999 and cannot both be zero" >&2
  exit 2
fi

aws elbv2 modify-listener \
  --region "$AWS_REGION" \
  --listener-arn "$ALB_LISTENER_ARN" \
  --default-actions "Type=forward,ForwardConfig={TargetGroups=[{TargetGroupArn=$CHAMPION_TG_ARN,Weight=$champion_weight},{TargetGroupArn=$CANDIDATE_TG_ARN,Weight=$candidate_weight}]}" \
  >/dev/null

echo "ALB weights updated: champion=$champion_weight candidate=$candidate_weight"
