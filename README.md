# Weather Prediction MLOps

Hệ thống dự báo thời tiết Hà Nội lúc 12:00, vận hành theo workflow MLOps gồm training, model release, canary deployment, prediction serving và monitoring chất lượng model.

## 1. Kiến trúc và workflow

```text
Airflow training DAG
  -> candidate model + MLflow registry + release_id
  -> Jenkins build/deploy image
  -> EC2 Docker Compose
  -> AWS ALB weighted routing
  -> Champion API / Canary API
  -> prediction logs
  -> Airflow monitoring DAG
  -> Parquet Data Lake + DuckDB + Evidently
  -> Prometheus / Pushgateway / Grafana / Alertmanager
```

Hai DAG chính:

- `weather_7d_training`: ingest, validate, build feature, train, evaluate và register candidate. DAG không tự động promote model.
- `weather_7d_monitoring`: lấy prediction log, thu thập actual weather data, tính performance, phát hiện drift và ghi monitoring artifact.

Model release được định danh bằng `release_id`. Monitoring chỉ tính dữ liệu thuộc release đang ở production. Dữ liệu và artifact của release cũ không bị xóa.

## 2. Yêu cầu môi trường

Local development:

- Python 3.12+
- Docker Engine và Docker Compose v2
- Git

AWS/EC2 deployment:

- AWS CLI
- Docker Compose trên EC2
- IAM Instance Profile cho EC2
- S3 bucket cho artifact và Data Lake
- Amazon ECR repository cho API image
- AWS ALB nếu muốn chạy canary routing

Không lưu AWS access key trong source code hoặc Docker image. Ưu tiên IAM Instance Profile trên EC2 và Jenkins Credentials/IAM role cho Jenkins.

## 3. Cấu hình môi trường

```bash
cp .env.example .env
chmod 600 .env
```

Tối thiểu cần cập nhật:

```dotenv
AWS_DEFAULT_REGION=us-east-1
S3_STORAGE_BUCKET=<your-s3-bucket>
S3_FEATURE_BUCKET=<your-s3-bucket>
MLFLOW_ARTIFACT_ROOT=s3://<your-s3-bucket>/weather-7d/mlflow-artifacts
AIRFLOW_DB_PASSWORD=<strong-password>
MLFLOW_DB_PASSWORD=<strong-password>
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=<strong-password>
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=<strong-password>
```

Nếu cần Telegram alert, cập nhật `TELEGRAM_BOT_TOKEN` và `TELEGRAM_CHAT_ID`. Không commit file `.env`.

## 4. Khởi động local stack

### 4.1. Build image

```bash
docker compose build mlflow airflow-init weather-api weather-api-canary
```

`weather-api` đọc model từ thư mục `models/` được mount read-only. Cần có `models/seven_day_production/production_manifest.json` trước khi API có thể ready.

### 4.2. Khởi động database, MLflow và Airflow

```bash
docker compose up -d postgres-airflow mlflow
docker compose run --rm airflow-init
docker compose up -d airflow-webserver airflow-scheduler
docker compose ps
docker compose logs --tail=100 airflow-scheduler
```

Các giao diện local:

| Service | URL |
|---|---|
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Pushgateway | http://localhost:9091 |
| Alertmanager | http://localhost:9093 |
| Grafana | http://localhost:3000 |

## 5. Chạy training DAG

```bash
docker compose exec -T airflow-scheduler airflow dags unpause weather_7d_training
docker compose exec -T airflow-scheduler airflow dags trigger weather_7d_training
docker compose exec -T airflow-scheduler airflow dags list-runs --dag-id weather_7d_training --output table
```

Sau khi DAG thành công, kiểm tra candidate release:

```bash
jq -r '.release_id' models/seven_day_production/registry_manifest.json
ls -la models/seven_day_production/releases/
```

Training DAG tạo candidate và registry artifacts. Model chưa trở thành production cho tới khi chạy promotion gate.

## 6. Promote model trong local demo

Chỉ chạy sau khi training và evaluation artifacts đã tồn tại:

```bash
docker compose exec -T airflow-scheduler \
  python -m src.production.cli_promote \
  --evaluation-manifest /opt/airflow/models/seven_day_production/evaluation_manifest.json \
  --registry-manifest /opt/airflow/models/seven_day_production/registry_manifest.json \
  --output-dir /opt/airflow/models/seven_day_production \
  --mlflow-tracking-uri http://mlflow:5000
```

Kiểm tra production release và reset metric:

```bash
jq -r '.release_id' models/seven_day_production/production_manifest.json
jq -r '.stage' models/seven_day_production/production_manifest.json
docker compose exec -T airflow-scheduler \
  python -m src.production.cli_monitor_reset \
  --production-manifest /opt/airflow/models/seven_day_production/production_manifest.json \
  --pushgateway-url http://pushgateway:9091
```

## 7. Khởi động API và monitoring

```bash
docker compose up -d prediction-logs-init weather-api weather-api-canary pushgateway prometheus alertmanager grafana
curl http://localhost:8000/ready
curl http://localhost:8001/ready
curl http://localhost:8000/api/v1/model/info
curl http://localhost:8001/api/v1/model/info
```

Prediction request cần tối thiểu 8 observation liên tiếp lúc 12:00. Dùng simulation script để tạo đúng payload:

```bash
python3 simulation/generate_prediction_logs.py \
  --url http://localhost:8000/api/v1/predict \
  --count 35
```

Prediction log production nằm trong Docker volume tại `/opt/airflow/project/monitoring/predictions/api_predictions.jsonl`. Canary ghi riêng vào `api_predictions_canary.jsonl`.

## 8. Kiểm thử ALB canary routing

```bash
AWS_REGION=<region> \
ALB_LISTENER_ARN=<listener-arn> \
CHAMPION_TG_ARN=<champion-target-group-arn> \
CANDIDATE_TG_ARN=<candidate-target-group-arn> \
./scripts/set_alb_weights.sh 90 10
```

```bash
python3 simulation/test_alb_routing.py \
  --url http://<ALB_DNS_NAME>/api/v1/predict \
  --count 1000 \
  --delay-seconds 0.02
```

Kết quả được ghi tại `simulation/alb_routing_responses.jsonl`. Tỷ lệ quan sát là gần đúng theo traffic weight.

## 9. Chạy monitoring DAG

Chỉ trigger sau khi có prediction log của production release:

```bash
docker compose exec -T airflow-scheduler airflow dags trigger weather_7d_monitoring
docker compose exec -T airflow-scheduler airflow dags list-runs --dag-id weather_7d_monitoring --output table
docker compose logs -f airflow-scheduler
```

Monitoring DAG:

1. Đọc `production_manifest.json` và lấy `release_id`.
2. Lọc prediction logs theo release hiện tại.
3. Gọi actual weather data.
4. Tạo `predictions.parquet`, `actuals.parquet` và `monitoring_dataset.parquet`.
5. Ghi Data Lake theo `event_date` và `release_id`.
6. Tính performance, retraining decision và Evidently drift.
7. Push metrics lên Pushgateway.

Kiểm tra artifact và metric:

```bash
docker compose exec -T airflow-scheduler cat /opt/airflow/monitoring/performance_report.json
docker compose exec -T airflow-scheduler cat /opt/airflow/monitoring/retraining_decision.json
docker compose exec -T airflow-scheduler ls -la /opt/airflow/project/monitoring/evidently/
curl http://localhost:9091/metrics | grep weather_model
```

## 10. Data Lake, Parquet và DuckDB

Monitoring Parquet được lưu theo prefix:

```text
s3://<bucket>/weather-7d/bronze/event_date=<date>/release=<release_id>/
```

Query monitoring dataset bằng DuckDB:

```bash
docker compose exec -T airflow-scheduler \
  python -m src.production.cli_query \
  --path s3://<bucket>/weather-7d/bronze/event_date=<date>/release=<release_id>/dataset/monitoring_dataset.parquet \
  --region "$AWS_DEFAULT_REGION" \
  --sql "SELECT model_release_id, COUNT(*) AS rows FROM lake_data GROUP BY 1"
```

DuckDB dùng AWS credential chain. IAM role cần quyền tối thiểu như `s3:GetObject`, `s3:PutObject` và `s3:ListBucket` trên bucket/prefix cần dùng.

## 11. Jenkins pipelines

### `Jenkinsfile.application`

Dùng khi code thay đổi: build API image, chạy test, push image lên ECR, SSH vào EC2, pull source và chạy `scripts/deploy_application.sh`. Pipeline không xóa database hoặc Docker volume.

### `Jenkinsfile`

Dùng cho model canary:

- `DEPLOY_CANARY`: build/push image và deploy candidate.
- `SET_WEIGHT`: cập nhật Champion/Candidate weight trên ALB.
- `PROMOTE`: promote release, archive manifest, chuyển candidate thành champion và reset monitoring metrics.
- `ROLLBACK`: đưa traffic về champion.

Jenkins credentials:

```text
aws-region
weather-app-host
weather-app-user
weather-app-dir
weather-ecr-registry
weather-ecr-repository
weather-app-ssh-key
```

## 12. Healthy baseline demo

```bash
python3 simulation/restore_healthy_baseline.py \
  --url http://localhost:8000/api/v1/predict \
  --count 35
```

Script tạo synthetic actual outcomes gần với prediction. Nó không xóa prediction logs hoặc dữ liệu production cũ. Đây là fixture cho demo, không phải bằng chứng chất lượng model thực tế.

## 13. Kiểm tra nhanh toàn bộ hệ thống

```bash
docker compose ps
curl http://localhost:8000/ready
curl http://localhost:8000/api/v1/model/info
curl http://localhost:9090/-/ready
curl http://localhost:3000/api/health
curl http://localhost:9091/metrics | grep weather_model
```

Kiểm tra source và test local:

```bash
python3 -m compileall -q src airflow/dags simulation
pytest -q tests/test_monitoring_dataset.py tests/test_registry_aliases.py tests/test_production_features.py
git diff --check
```

## 14. EC2 deployment thủ công

```bash
cd ~/weather-prediction-Machine-Learning
git pull --ff-only origin main

AWS_REGION=<region> \
ECR_REGISTRY=<account>.dkr.ecr.<region>.amazonaws.com \
ECR_REPOSITORY=<repository> \
IMAGE_TAG=<commit-sha> \
bash scripts/deploy_application.sh
```

Kiểm tra sau deploy:

```bash
docker compose ps
docker compose logs --tail=100 weather-api
docker compose logs --tail=100 airflow-scheduler
curl http://localhost:8000/ready
curl http://localhost:8000/api/v1/model/info
```

Nếu Airflow dependency thay đổi:

```bash
docker compose build airflow-init
docker compose run --rm airflow-init
```

## 15. Troubleshooting và an toàn dữ liệu

API chưa ready:

```bash
docker compose logs --tail=200 weather-api weather-api-canary
ls -l models/seven_day_production/production_manifest.json
```

ALB timeout hoặc 503:

```bash
curl -v http://<ALB_DNS_NAME>/ready
aws elbv2 describe-target-health --region <region> --target-group-arn <target-group-arn>
```

Monitoring không có dữ liệu:

```bash
docker compose exec -T airflow-scheduler wc -l /opt/airflow/project/monitoring/predictions/api_predictions.jsonl
docker compose exec -T airflow-scheduler python -c "import json; p='/opt/airflow/models/seven_day_production/production_manifest.json'; print(json.load(open(p))['release_id'])"
```

Nếu log không có `model_release_id` hoặc không trùng production manifest, DAG sẽ không thể tính đúng release-scoped monitoring.

Không chạy lệnh sau trong demo production nếu chưa có backup:

```bash
docker compose down -v
```

Lệnh này xóa Docker volumes, bao gồm prediction logs và metadata database.
