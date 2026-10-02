# Prediction log simulation

This helper calls the running production API with synthetic historical
observations. The API itself writes the canonical log to the shared Docker
volume at `monitoring/predictions/api_predictions.jsonl`.

Run from the repository root:

```bash
python simulation/generate_prediction_logs.py \
  --url http://localhost:8000/api/v1/predict \
  --count 35
```

For an EC2-hosted API, use the public API URL instead:

```bash
python simulation/generate_prediction_logs.py \
  --url http://<EC2_PUBLIC_IP>:8000/api/v1/predict \
  --count 35
```

The default timestamps are in the past, so the Open-Meteo actuals collector
can resolve their seven forecast horizons. At least 30 requests are generated
because the default Evidently drift window uses 30 current feature snapshots.

## Test ALB canary routing

This script sends fresh requests through the ALB and counts the `model_track`
returned by the API. It does not persist cookies, so it is suitable for
checking a weighted split such as champion 90 / canary 10.

```bash
python3 simulation/test_alb_routing.py \
  --url http://<ALB_DNS_NAME>/api/v1/predict \
  --count 1000
```

The result is written to `simulation/alb_routing_responses.jsonl`. Observed
counts are approximate, not an exact per-request guarantee.
