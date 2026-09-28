# 7-Day Production MLOps Contract

The production scope is the seven-day forecast. The API does not train models,
select candidates, or evaluate test data. The lifecycle is intentionally
sequential:

```text
train candidates -> evaluate candidates -> publish production manifest -> serve
```

## Training

```bash
python -m src.production.cli_train \
  --output-dir models/seven_day_production
```

Training fits all candidate models using only the training period and writes:

- `candidate_manifest.json`
- candidate models for temperature and rain for horizons 1..7

The command does not publish a serving model.

## Evaluation and promotion

Run only after training succeeds:

```bash
python -m src.production.cli_evaluate \
  --candidate-manifest models/seven_day_production/candidate_manifest.json \
  --output-dir models/seven_day_production
```

Evaluation loads the candidates, selects models using validation data, tunes rain
thresholds on validation data, evaluates the selected models once on the test
period, and writes:

- `evaluation_metrics.csv`
- `evaluation_manifest.json`

Promotion is a separate gate:

```bash
python -m src.production.cli_promote \
  --evaluation-manifest models/seven_day_production/evaluation_manifest.json \
  --output-dir models/seven_day_production
```

Only this command creates `production_manifest.json` and `champion_manifest.json`.
The default gate requires test RMSE <= 4.0°C and rain F1 >= 0.55.

Only the production manifest is allowed to be consumed by the API.

## Serving

```bash
uvicorn src.production.serving:app --host 0.0.0.0 --port 8000
```

The Docker Compose API mounts `models/` read-only. Run training and evaluation
before starting the API on a clean machine; the image itself never trains and
does not bake model binaries into the container.

## Tracking and monitoring

To record pipeline evidence in MLflow:

```bash
python -m src.production.cli_train \
  --output-dir models/seven_day_production \
  --mlflow-tracking-uri http://localhost:5000

python -m src.production.cli_evaluate \
  --candidate-manifest models/seven_day_production/candidate_manifest.json \
  --output-dir models/seven_day_production \
  --mlflow-tracking-uri http://localhost:5000
```

The API exposes Prometheus metrics at `/metrics`. The Compose Prometheus service
scrapes this endpoint and loads the rules in `monitoring/alert_rules.yml`.

The Airflow monitoring DAG pushes batch model metrics to Pushgateway. Prometheus
then evaluates alerts for degraded RMSE, low rain F1, drift, and retraining
requirements. API alerts and model alerts therefore use the same Alertmanager
route.

Airflow DAG definitions are in `airflow/dags/`; the training DAG keeps candidate
training and evaluation as separate sequential tasks, while the monitoring DAG
checks the published production manifest, calculates performance, and writes a
machine-readable retraining decision from prediction/actual data.

Endpoints:

- `GET /health`
- `GET /ready`
- `GET /api/v1/model/info`
- `POST /api/v1/predict`

The request must contain at least eight chronological noon observations. This is
the minimum required to construct the seven-day lag and rolling features without
using future targets.

## Artifact promotion rule

`candidate_manifest.json` is an intermediate artifact. A model becomes usable by
the API only when `production_manifest.json` exists and has
`stage: production`. This creates a clear promotion boundary for a future MLflow
registry alias (`candidate` -> `champion`).
