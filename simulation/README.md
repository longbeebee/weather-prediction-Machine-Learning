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
