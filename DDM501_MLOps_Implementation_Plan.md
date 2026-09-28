# DDM501 Final Project - MLOps Implementation Plan

> Project: End-to-End Weather Forecasting ML System  
> Course: DDM501 - AI in Production: From Models to Systems  
> Purpose: Master execution plan for Codex and the project team  
> Delivery window: 4 weeks

---

## 1. Project Goal

Build, test, deploy, monitor, and document a complete weather forecasting ML system.

The core lifecycle is:

```text
Weather data
  -> ingestion and validation
  -> preprocessing and feature engineering
  -> model training and evaluation
  -> MLflow experiment tracking and model registry
  -> REST API serving
  -> Docker Compose deployment
  -> Prometheus, Grafana, and alerts
  -> drift and performance evaluation
  -> controlled retraining
```

The primary goal is a reliable working implementation and live demo. Kubernetes, Jenkins, and advanced AWS services are extensions, not substitutes for the rubric-required components.

---

## 2. Mandatory Scope and Priority

### P0 - Required for the rubric

- Problem definition, users, use cases, constraints, and measurable success criteria.
- Modular data, feature, training, evaluation, and model-selection pipeline.
- Data validation, versioning, logging, and error handling.
- Multiple model experiments with MLflow parameters, metrics, and artifacts.
- Versioned REST API with OpenAPI documentation and error handling.
- Secure multi-stage Dockerfile.
- Full local multi-service deployment with Docker Compose and health checks.
- Prometheus application and ML metrics.
- Grafana dashboard and meaningful alert rules.
- Unit, integration, data-quality, and model-validation tests.
- At least 80% meaningful test coverage for core application modules.
- GitHub Actions pipeline for lint, tests, build, and deployment validation.
- SHAP-based explainability, fairness or segment analysis, privacy, and ethics discussion.
- `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, API documentation, and operating guide.

### P1 - Strong production extensions

- S3 for datasets, model artifacts, and reports.
- PostgreSQL or RDS for MLflow and prediction metadata.
- ECR for container images.
- EC2 for compute.
- Evidently for drift and performance reports.
- Champion/candidate model aliases and validation gate.
- Telegram alerts through Alertmanager or a webhook adapter.

### P2 - Only after all P0 items work

- Single-node k3s on EC2.
- Champion and candidate deployments.
- Canary or separate candidate endpoint.
- HPA, readiness, liveness, and startup probes.
- Jenkins deployment pipeline.
- Automatic rollback.
- Terraform, Secrets Manager, SQS/Kinesis, and GitOps.

---

## 3. Architecture Decision

### Local and grading baseline

```text
Developer / GitHub Actions
          |
          v
   Docker Compose
   +-------------------------------+
   | FastAPI prediction service    |
   | MLflow tracking server        |
   | PostgreSQL                    |
   | Airflow                       |
   | Prometheus + Alertmanager     |
   | Grafana                       |
   | Evidently job                 |
   +-------------------------------+
          |
          v
   Local/S3-compatible artifacts
```

### AWS target

```text
GitHub -> GitHub Actions -> ECR -> EC2
                                  |
                            Docker Compose
                            or optional k3s
                                  |
                  +---------------+---------------+
                  |               |               |
                  v               v               v
                 S3              RDS             ECR
            data/artifacts   metadata/results   images
```

Decision: use EC2 for compute and AWS managed services for important persistent state. The first deployable version must remain runnable locally through Docker Compose.

---

## 4. Repository Structure

```text
weather-mlops/
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │   └── preprocessing.py
│   ├── features/
│   │   └── build_features.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── model_selection.py
│   │   └── registry.py
│   ├── serving/
│   │   ├── app.py
│   │   ├── schemas.py
│   │   ├── model_loader.py
│   │   └── metrics.py
│   ├── monitoring/
│   │   ├── drift.py
│   │   ├── performance.py
│   │   └── explainability.py
│   └── common/
│       ├── config.py
│       └── logging.py
├── airflow/dags/
│   ├── weather_training_pipeline.py
│   └── weather_monitoring_pipeline.py
├── configs/
│   ├── base.yaml
│   ├── training.yaml
│   └── production.yaml
├── data/
│   ├── raw/.gitkeep
│   ├── processed/.gitkeep
│   └── reference/.gitkeep
├── monitoring/
│   ├── prometheus.yml
│   ├── alert_rules.yml
│   ├── alertmanager.yml
│   └── grafana/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── data/
│   └── model/
├── scripts/
├── docs/
│   ├── images/
│   ├── demo-script.md
│   └── operations-guide.md
├── kubernetes/                 # P2 only
├── terraform/                  # P2 only
├── .github/workflows/
│   └── ci-cd.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── ARCHITECTURE.md
└── CONTRIBUTING.md
```

Do not commit raw datasets, model binaries, secrets, `.env`, local databases, or generated monitoring data.

---

## 5. Rules for Codex

Codex must follow these rules for every phase:

1. Inspect the current repository and reuse working code before editing.
2. State the files and behavior to be changed before implementation.
3. Preserve existing algorithms and evaluation logic unless a verified defect requires a change.
4. Implement only the current phase. Do not implement future phases.
5. Do not hard-code paths, credentials, URLs, thresholds, hyperparameters, or environment-specific values.
6. Use type hints, docstrings, structured logging, and explicit error handling.
7. Add or update tests for every material behavior change.
8. Run relevant lint, test, build, and smoke-test commands after editing.
9. Fix failures caused by the change before stopping.
10. Do not claim success without showing the command result.
11. Preserve unrelated user changes in the repository.
12. Summarize changed files, commands run, results, limitations, and the next recommended phase.
13. Stop when the phase acceptance criteria are met and wait for approval.

Use small, meaningful commits such as:

```text
refactor: modularize training pipeline
feat: add MLflow experiment tracking
test: add prediction API integration tests
ops: add Prometheus and Grafana configuration
docs: document architecture trade-offs
```

---

## 6. Implementation Phases

### Phase 0 - Audit and requirements baseline

#### Tasks

- Inventory current source code, datasets, notebooks, models, tests, and commands.
- Identify reusable code, duplicate logic, hard-coded configuration, and missing dependencies.
- Create a traceable requirements table containing:
  - functional requirements;
  - non-functional requirements;
  - business, model, and system success metrics;
  - assumptions, constraints, risks, and out-of-scope items.
- Record initial baseline metrics for every existing model.
- Define the demo scenario and sample API request.

#### Required success metrics

| Level | Metric | Initial target |
|---|---|---|
| Model | RMSE/MAE | Better than the agreed baseline |
| Model | R2 | Reported where meaningful |
| API | p95 latency | Under 500 ms on the demo environment |
| API | Error rate | Under 1% for valid requests |
| Reliability | Service availability during demo | 100% |
| Quality | Core test coverage | At least 80% |
| Operations | Critical alert delivery | Under 2 minutes |

Targets may be adjusted after the baseline is measured, but the reason must be documented.

#### Acceptance criteria

- [ ] Repository audit is written in `docs/audit.md`.
- [ ] Requirements and targets are documented in `README.md` or `docs/requirements.md`.
- [ ] Baseline command and results are reproducible.
- [ ] No production code is changed unnecessarily.

---

### Phase 1 - Standardize the ML pipeline

#### Tasks

- Refactor ingestion, validation, preprocessing, features, training, evaluation, and selection into modules.
- Move paths, thresholds, model lists, random seeds, and hyperparameters into YAML.
- Create one common training interface for all model families.
- Produce a model artifact, `metrics.json`, evaluation plots, and best-model summary.
- Add deterministic seeds, structured logs, and actionable error messages.
- Add unit tests for configuration, validation, preprocessing, features, metrics, and selection.

#### Required command

```bash
python -m src.training.train --config configs/training.yaml
```

#### Acceptance criteria

- [ ] Existing models still train successfully.
- [ ] At least two models use the same pipeline interface.
- [ ] RMSE and MAE are produced; R2 is produced where applicable.
- [ ] Important configuration is not hard-coded.
- [ ] Tests pass and outputs are reproducible with the same seed.
- [ ] MLflow, cloud, Airflow, and Kubernetes are not implemented yet.

---

### Phase 2 - MLflow tracking and registry

#### Tasks

- Run MLflow locally first, backed by PostgreSQL when using Compose.
- Create one parent run per pipeline execution and one child run per model experiment.
- Log dataset version/hash, feature version, code commit, parameters, metrics, plots, and model artifacts.
- Register the best new model as `candidate`.
- Support aliases: `champion`, `candidate`, and `previous`.
- Compare candidate against champion through a validation gate.
- Never replace champion automatically only because training completed.

#### Validation gate

Minimum checks:

- candidate RMSE and MAE versus champion;
- inference latency;
- model size;
- schema compatibility;
- model-load smoke test;
- configurable minimum improvement threshold.

#### Acceptance criteria

- [ ] MLflow UI shows parameters, metrics, dataset identity, plots, and models.
- [ ] Registered candidate can be loaded from the registry.
- [ ] A failed candidate is rejected with a recorded reason.
- [ ] Champion is unchanged until an explicit promotion decision.

---

### Phase 3 - FastAPI model serving

#### Required endpoints

```text
GET  /health
GET  /ready
GET  /api/v1/model/info
POST /api/v1/predict
GET  /metrics
```

#### Tasks

- Load `models:/weather-forecast@champion`, not a hard-coded local pickle path.
- Validate requests and return stable response schemas.
- Include `request_id`, prediction, model version, model alias, and latency.
- Implement readiness as false until the model is loaded and usable.
- Store or log prediction metadata without storing unnecessary personal data.
- Provide OpenAPI examples and documented error responses.

#### Acceptance criteria

- [ ] Valid prediction returns HTTP 200 and the complete response schema.
- [ ] Invalid data returns a useful HTTP 4xx response.
- [ ] Model unavailable returns a controlled HTTP 5xx response.
- [ ] `/health`, `/ready`, and `/metrics` behave correctly.
- [ ] Integration tests cover success and failure cases.

---

### Phase 4 - Docker and Docker Compose

#### Services

- `weather-api`
- `postgres`
- `mlflow`
- `airflow-webserver`
- `airflow-scheduler`
- `prometheus`
- `alertmanager`
- `grafana`
- optional one-shot Evidently report service

#### Tasks

- Create an optimized multi-stage Dockerfile with a non-root runtime user.
- Pin dependencies and add a `.dockerignore`.
- Add health checks, named volumes, isolated networks, restart policy, and resource guidance.
- Put sample values only in `.env.example`; never commit real secrets.
- Ensure service startup order depends on health, not fixed sleeps.

#### Required commands

```bash
docker compose config
docker compose build
docker compose up -d
docker compose ps
```

#### Acceptance criteria

- [ ] One command starts the full local system.
- [ ] All required services become healthy.
- [ ] API, MLflow, Airflow, Prometheus, and Grafana are reachable.
- [ ] Data survives container restart through volumes.
- [ ] A clean-machine setup is documented and repeatable.

---

### Phase 5 - Airflow orchestration

Create two separate DAGs.

#### Training DAG

```text
ingest
  -> validate
  -> preprocess
  -> build_features
  -> train_models
  -> evaluate
  -> compare_with_champion
  -> register_candidate
```

#### Monitoring DAG

```text
collect_actuals
  -> join_predictions_with_actuals
  -> calculate_performance
  -> run_drift_checks
  -> decide_retraining
```

#### Rules

- Tasks must be idempotent and retry-safe.
- Large data must be passed by path/object reference, not Airflow XCom.
- Failed validation must stop downstream training.
- Retraining requires a documented policy combining schedule, enough new data, drift, or performance degradation.

#### Acceptance criteria

- [ ] Both DAGs load without import errors.
- [ ] Training DAG completes successfully on sample data.
- [ ] Failure and retry behavior are demonstrated.
- [ ] Monitoring DAG produces a machine-readable decision.

---

### Phase 6 - Monitoring, drift, and alerting

#### Prometheus metrics

- request count and status;
- request duration histogram and p95 latency;
- prediction count and prediction latency;
- model load failures;
- active model version/alias;
- latest model RMSE and MAE;
- drift score and drifted feature count;
- missing/invalid input count;
- Airflow DAG success/failure where available;
- CPU and memory through suitable exporters.

Avoid unbounded metric labels such as raw `request_id`, user input, timestamp, or exact prediction value.

#### Grafana dashboard sections

- API health: traffic, p95 latency, errors.
- Model health: version, RMSE, MAE, prediction distribution.
- Data health: drift, missing data, feature distributions.
- Infrastructure: CPU, RAM, container health.

#### Alerts

- API high error rate.
- p95 latency above target.
- service or container down.
- model performance degradation.
- data drift detected.
- Airflow pipeline failure.

#### Acceptance criteria

- [ ] Evidently produces HTML and JSON outputs.
- [ ] Prometheus successfully scrapes the API.
- [ ] Grafana dashboard loads with meaningful data.
- [ ] At least one alert is triggered and resolved in a controlled demo.
- [ ] Telegram is optional; Alertmanager rules are mandatory.

---

### Phase 7 - Testing and GitHub Actions CI/CD

#### Test categories

- Unit tests for business and ML pipeline logic.
- API integration tests.
- Data-quality tests for schema, ranges, nulls, duplicates, and time ordering.
- Model tests for minimum quality, schema, serialization, determinism, and latency.
- Docker Compose smoke tests where practical.

#### Required CI stages

```text
checkout
  -> dependency install and cache
  -> formatting/lint
  -> type check
  -> unit + data + model tests
  -> integration tests
  -> coverage gate
  -> Docker build
  -> container scan
  -> Compose/config validation
  -> optional push/deploy on protected branch
```

#### Acceptance criteria

- [ ] GitHub Actions workflow is in `.github/workflows/`.
- [ ] Pull requests cannot pass when lint, tests, or coverage fail.
- [ ] Core coverage is at least 80% and meaningful.
- [ ] CI contains no plaintext credentials.
- [ ] Deployment steps are branch-protected and environment-aware.

Note: Jenkins may be added later for demonstration, but it does not replace the rubric-required GitHub Actions workflow.

---

### Phase 8 - Responsible AI

#### Tasks

- Implement global and local SHAP explanations for the selected forecasting model.
- Analyze error by meaningful segments such as location, season, temperature band, or forecast horizon.
- Define what fairness means for this forecasting context; do not force demographic fairness metrics when demographic groups do not exist.
- Identify high-error or under-represented segments and propose mitigation.
- Document collected fields, retention, access control, secrets, logging redaction, and deletion policy.
- Document misuse, incorrect forecast risk, over-reliance, data ownership, and operational safeguards.

#### Acceptance criteria

- [ ] SHAP summary plot and at least one local explanation are reproducible.
- [ ] Segment performance table is included.
- [ ] Limitations and mitigation actions are explicit.
- [ ] Privacy and ethics sections are present in the documentation.

---

### Phase 9 - AWS deployment

#### Target resources

- EC2 for application compute.
- S3 for raw/processed data, MLflow artifacts, and Evidently reports.
- RDS PostgreSQL for MLflow and operational metadata.
- ECR for container images.
- IAM roles with least privilege.
- Security groups exposing only necessary ports.
- Optional Secrets Manager for production credentials.

#### Deployment order

1. Create persistent AWS resources.
2. Configure IAM and network rules.
3. Build and push images to ECR.
4. Deploy the already-working Compose stack to EC2.
5. Connect MLflow to RDS and S3.
6. Run migrations and health checks.
7. Execute an end-to-end smoke test.
8. Back up, restart, and verify recovery.

#### Acceptance criteria

- [ ] No long-lived AWS key is committed or baked into an image.
- [ ] EC2 can be replaced without losing important artifacts or metadata.
- [ ] Public access is limited to required application endpoints.
- [ ] API prediction, MLflow artifact logging, and monitoring work on AWS.
- [ ] Cost and shutdown instructions are documented.

---

### Phase 10 - Optional k3s, candidate deployment, and rollback

Only start after Phases 0-9 are stable.

#### Tasks

- Install single-node k3s on EC2.
- Create namespace, ConfigMaps, Secrets, Deployments, Services, Ingress, and probes.
- Deploy champion and candidate separately.
- Use a separate candidate endpoint first; add weighted routing only if time permits.
- Add smoke-test-driven promotion and rollback.
- Add HPA as a demonstrable concept, while documenting the limitation of a single node.

#### Acceptance criteria

- [ ] Champion and candidate are independently observable.
- [ ] Failed readiness prevents traffic.
- [ ] Candidate failure does not interrupt champion.
- [ ] Promotion updates aliases and deployment state consistently.
- [ ] Rollback restores the previous working version.

---

## 7. Definition of Done for Every Phase

A phase is complete only when all conditions below are true:

- [ ] Requested behavior is implemented.
- [ ] Relevant tests pass.
- [ ] New configuration is documented in `.env.example` or YAML.
- [ ] No secret or large generated artifact is committed.
- [ ] Logs and errors are understandable.
- [ ] Commands and outputs used for verification are recorded.
- [ ] Documentation reflects the actual implementation.
- [ ] Changes are committed with a meaningful message.
- [ ] Known limitations are recorded.
- [ ] The next phase has not been implemented early.

---

## 8. Four-Week Delivery Plan

| Week | Main outcome | Required demo |
|---|---|---|
| 1 | Requirements, architecture, modular ML pipeline, tests | Reproducible training and baseline comparison |
| 2 | MLflow, registry, FastAPI, Docker | Registered model served through documented API |
| 3 | Docker Compose, Airflow, Prometheus, Grafana, alerts | End-to-end local lifecycle and failure alert |
| 4 | CI/CD, Responsible AI, AWS, documentation, rehearsal | Working cloud demo, explanations, rollback/recovery story |

Freeze optional features early if any P0 rubric item is incomplete.

---

## 9. Final Documentation Checklist

### `README.md`

- Problem and business context.
- Architecture overview.
- Prerequisites and quick start.
- Training and prediction examples.
- Service URLs and credentials setup.
- Testing and CI instructions.
- Monitoring guide.
- Troubleshooting.
- Limitations and future work.

### `ARCHITECTURE.md`

- Context, container, deployment, and data-flow diagrams.
- Component responsibilities.
- Online prediction flow.
- Training and retraining flow.
- Failure cases and recovery.
- Technology decisions and trade-offs for scalability, cost, and complexity.

### `CONTRIBUTING.md`

- Named team roles and responsibilities.
- Branching, review, and commit conventions.
- Local quality checks.
- Evidence of meaningful contributions by every member.

### Demo preparation

- 15-20 minute presentation plus 10 minute Q&A.
- All team members participate.
- Prepare a deterministic demo dataset and requests.
- Pre-pull images and verify services before presentation.
- Keep screenshots or a short backup recording for recovery only.
- Demonstrate one normal flow and one controlled failure/alert flow.

---

## 10. Master Prompt for Starting a Phase

Copy the following prompt into Codex and replace the placeholders.

```text
You are acting as a senior MLOps engineer working on the DDM501 final project.

Read DDM501_MLOps_Implementation_Plan.md completely.
The current phase is: <PHASE NUMBER AND NAME>.

Implement only this phase. Do not implement future phases.

Before editing:
1. Inspect the repository, git status, current tests, and relevant documentation.
2. Identify reusable existing components.
3. Briefly state the proposed file-level changes and risks.

During implementation:
1. Preserve working ML logic unless a verified defect requires a change.
2. Keep configuration outside code.
3. Add type hints, structured logging, error handling, and tests.
4. Preserve unrelated user changes.
5. Never commit secrets, datasets, generated artifacts, or credentials.

After implementation:
1. Run all phase-specific acceptance commands.
2. Run relevant lint, tests, coverage, build, and smoke tests.
3. Fix failures caused by the change.
4. Update documentation to match reality.
5. Summarize changed files, verification results, limitations, and the next recommended phase.
6. Stop and wait for approval. Do not implement the next phase.

Phase requirements and acceptance criteria are defined in the implementation plan.
```

---

## 11. Recommended First Codex Task

Start with Phase 0, not AWS or Kubernetes.

```text
Execute Phase 0 - Audit and requirements baseline from
DDM501_MLOps_Implementation_Plan.md.

Do not refactor or add infrastructure yet. Inspect the existing weather forecasting
repository, document reusable components and gaps, reproduce the current baseline,
and create the requirements/success-metrics documentation required by Phase 0.

Run only safe read-only or baseline commands until the audit is complete. Then make
only the documentation and minimal reproducibility changes required by Phase 0.
Stop after all Phase 0 acceptance criteria are verified.
```

---

## 12. Final Release Gate

The project is ready for submission only if:

- [ ] A clean checkout can start the required stack using documented commands.
- [ ] Training, registry, serving, monitoring, and evaluation form one traceable lifecycle.
- [ ] Docker Compose, GitHub Actions, tests, monitoring, and Responsible AI are complete.
- [ ] The API and dashboards work during a rehearsal.
- [ ] The report describes only features that actually work or clearly labels prototypes.
- [ ] Architecture decisions include cost, scalability, complexity, and failure trade-offs.
- [ ] Every team member can explain their code and architecture contribution.
- [ ] The live demo has a tested recovery path.

