# Loan‑Radar

*Predicting Micro‑Finance Loan Default Risk on Chameleon Cloud*

---

## Overview

**Loan‑Radar** is an end‑to‑end, cloud‑native machine‑learning system that predicts the probability that a borrower will default on a consumer loan.  Built for NYU’s *ECE‑GY 9183 – Machine‑Learning Systems*, the project demonstrates:

* **Medium‑scale data** – 1.9 M rows (≈ 2 GB) from the public LendingClub loan book.
* **Distributed training** – Ray Tune hyper‑parameter search logged to a self‑hosted MLflow server.
* **Fully containerised deployment** on the Chameleon Cloud KVM platform.
* **Monitored inference service** – FastAPI + Prometheus + Grafana with < 1 ms median latency.

---

## 1 · Value Proposition

| Stakeholder                                                    | Pain Point                                                               | ML Solution                                                                                                          | Business KPI                                                                          |
| -------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Credit‑risk officers at regional banks & micro‑finance lenders | Manual rule‑based underwriting is slow, subjective, and often inaccurate | **Real‑time REST API** returning **Low / Medium / High** default‑risk labels + feature‑level explanation in < 120 ms | • ↓ Default‑rate on new loans<br>• ↑ Application throughput<br>• ↓ Manual‑review cost |

---

## 2 · Dataset & External Assets

| Asset                               | Lineage / Licence                                   | Notes                                                                             |
| ----------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------- |
| **LendingClub 2007‑2018 loan book** | Public dataset mirrored on Kaggle – CC BY‑NC‑SA 4.0 | 2 GB of accepted‑loan records; target `loan_status` mapped to 3‑class risk level. |
| **scikit‑learn / XGBoost**          | BSD‑3 & Apache‑2.0                                  | XGBoost selected for tabular interpretability & speed.                            |

No proprietary data or closed models are used.

---

## 3 · System Architecture

```mermaid
graph LR
    subgraph Training
        A[Ingest CSVs] --> B[ETL Docker job\n(clean & feature engineering)]
        B --> C[Ray Tune HPO\n(XGBoost, 32 trials)]
        C --> D[MLflow Tracking]
        C --> E[Pick best model\n(F1 & AUC)]
        E --> F[Model Registry (MLflow)]
    end
    subgraph CI/CD Pipeline
        P[GitHub Action Trigger] --> Q[Ray Job Submit\n(re‑training)]
        Q --> D
        Q -->|buildx| R[Container Image\n(model‑server)]
        R --> S[Helm Upgrade --install\n(staging)]
    end
    subgraph Serving
        U[FastAPI + Uvicorn] -.-> V[Prometheus Metrics]
        U --> W[Grafana Dashboards]
        U --> X[Client (React form)]
    end
```

*Solid arrows = data / model artefacts · Dashed = monitoring hooks*

---

## 4 · Data Pipeline

1. **Extract / Load** – `docker-compose-etl.yaml` spins up a Python 3.11 container that downloads the raw CSV from Google Drive (via `gdown`), unzips, and stages to `/mnt/data/raw` (S3‑compatible object store).
2. **Transform** – `transform.py` processes the dataset in chunks, imputes missing values, one‑hot & label‑encodes categoricals, scales numerics, and writes partitioned Parquet files to `/mnt/data/processed`.
3. **Split** – `data‑split.py` performs a stratified 70 / 15 / 15 train‑val‑eval split and saves the CSVs to object storage.
4. **Streaming simulator** – `simulator.py` publishes JSON applicant events to Kafka topic **loan‑raw**; a consumer batches them for online evaluation.

Persistent block and object storage volumes are provisioned via Terraform and mounted on the VM.

---

## 5 · Model Training

| Item           | Setting                                                                    |
| -------------- | -------------------------------------------------------------------------- |
| **Algorithm**  | XGBoost (`binary:logistic`)                                                |
| **Key params** | `max_depth = 15`, `n_estimators = 150`, `scale_pos_weight = 6.4`           |
| **HPO**        | Ray Tune (ASHA) exploring depth, learning rate, subsampling over 32 trials |
| **Tracking**   | All runs logged to self‑hosted MLflow (Docker Compose)                     |

**Best model metrics**

| Metric             | Validation | Test |
| ------------------ | ---------- | ---- |
| F1 (macro)         | **0.78**   | 0.76 |
| ROC‑AUC            | **0.82**   | 0.81 |
| PR‑AUC (High‑risk) | **0.74**   | 0.72 |

---

## 6 · Deployment & DevOps

| Layer           | Tooling                    | Notes                                                                                      |
| --------------- | -------------------------- | ------------------------------------------------------------------------------------------ |
| **IaC**         | Terraform (HCL)            | Creates `m1.medium` KVM VM, attaches 100 GB block volume & floating IP.                    |
| **Config Mgmt** | Ansible                    | Installs Docker & Docker Compose.                                                          |
| **Containers**  | Docker Compose             | Services: ETL, MLflow, Ray‑head, Ray‑worker, FastAPI, Prometheus, Grafana.                 |
| **CI**          | GitHub Actions *(planned)* | Runs unit tests & builds images on push to `main`.                                         |
| **CD**          | Helm *(manual promotion)*  | `helm upgrade --install loan‑radar ./helm` deploys to `staging` → `canary` → `production`. |

---

## 7 · Serving & Monitoring

* **Inference API** – FastAPI under Gunicorn/Uvicorn (`/predict` POST) handling JSON or CSV batch payloads.
* **Performance**

  * **Median latency**: **0.79 ms**
  * **95th‑percentile latency**: **0.87 ms**
  * **Throughput**: **33,083 samples / sec**
    *(measured on 100‑sample micro‑benchmark)*
* **Observability** – Prometheus counters/histograms for prediction confidence & class frequency; Grafana dashboard visualises latency, error rate, and traffic.

```bash
curl -X POST https://<ip>/predict \
     -H "Content-Type: application/json" \
     -d '{"annual_inc": 50000, "int_rate": 13.5, ...}'
```

---

## 8 · Evaluation & Testing

| Suite                        | Tool                     | Status                                                                                                                                |
| ---------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Offline unit/slice tests** | `pytest`                 | ![6 passed](https://img.shields.io/badge/tests-6%20passed-brightgreen) ![2 failed](https://img.shields.io/badge/tests-2%20failed-red) |
| **Load test (staging)**      | Locust 50 users × 3 min  | p95 latency = 115 ms                                                                                                                  |
| **Online canary**            | Synthetic re‑play driver | Acceptance‑rate delta < ±2 %                                                                                                          |

<details><summary>Latest pytest summary</summary>

```text
===================== short test summary info =====================
FAILED tests/test_calibration.py::test_noisy_imp_features   F1 unexpectedly high with noise in important features: 0.373
FAILED tests/test_calibration.py::test_random_noise        F1 unexpectedly high on random noise: 0.517
==================== 2 failed, 6 passed in 8.83s =================
```

</details>

*Failing calibration tests are under investigation; they do **not** block deployment but raise a monitoring alert.*

---

## 9 · Quick‑Start (Local Demo)

```bash
# 1. Clone repo & build images
$ git clone https://github.com/ds28-ops/loan-default-risk-assessment.git \
  && cd loan-default-risk-assessment
$ docker compose -f docker-compose-fastapi.yaml up --build

# 2. Open docs & dashboards
→ Swagger:  http://localhost:8000/docs
→ Grafana:  http://localhost:3000  (login: admin / admin)
```

---

## Contributors

| Name                    | Role                                           |
| ----------------------- | ---------------------------------------------- |
| **Sampreeth Avvari**    | Data engineering · Offline evaluation          |
| **Dhruv Sridhar**       | Ray cluster ops · Experiment tracking          |
| **Barath Rama Shankar** | Backend API · Online evaluation · Architecture |

---

## License

Released under the [MIT License](LICENSE).
