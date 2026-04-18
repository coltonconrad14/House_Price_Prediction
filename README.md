# House Price Prediction

End-to-end machine learning scaffold for training and serving a house price regression model.

The primary training pipeline now uses live API feature candidates written to JSONL (not CSV).

## Project Structure

```text
.
├── data/
│   ├── raw/                # Optional legacy source files
│   └── processed/          # Live feature store + generated outputs
├── models/                 # Trained model artifacts (.joblib)
├── notebooks/              # Exploration notebooks
├── scripts/
│   ├── train.py            # Train + evaluate + save model
│   └── predict.py          # Batch inference from CSV
├── src/house_price_prediction/
│   ├── config.py           # Env-based settings
│   ├── data.py             # Data loading and split helpers
│   ├── features.py         # Preprocessing pipeline
│   ├── model.py            # Training, metrics, serialization
│   └── predict.py          # Inference helper
├── tests/
├── .env.example
├── pyproject.toml
└── requirements-dev.txt
```

## Environment Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

3. Copy environment defaults:

```bash
cp .env.example .env
```

## Data Expectations

- Default training file path: `data/processed/live_feature_store.jsonl`
- Default target column: `SalePrice`
- Configure via `.env` if your data uses different names or path overrides

Example `.env` values:

```env
RAW_DATA_PATH=data/processed/live_feature_store.jsonl
TARGET_COLUMN=SalePrice
MODEL_PATH=models/house_price_model.joblib
TEST_SIZE=0.2
RANDOM_STATE=42
TRAINING_MIN_ROWS=25
```

## Train

Populate the live feature store from API traffic first:

```bash
python scripts/bootstrap_training_data.py --base-url http://127.0.0.1:8000
```

```bash
python scripts/train.py --min-rows 25
```

Output:
- model saved at `models/house_price_model.joblib`
- printed evaluation metrics: MAE, RMSE, R2
- bootstrap metadata saved at `data/processed/live_feature_store.jsonl.meta.json`

For strict retraining readiness, enforce a minimum extracted row threshold and write timestamped snapshots:

```bash
python scripts/bootstrap_training_data.py --base-url http://127.0.0.1:8000 --min-output-rows 25 --snapshot-dir data/processed/snapshots
```

Bootstrap metadata now includes canonicalization diagnostics to support future feature work:
- `unknown_source_feature_keys`
- `rows_dropped_missing_required`
- `rows_with_invalid_target`

You can point Make bootstrap flows at any running API instance:

```bash
make bootstrap-data-snapshot BOOTSTRAP_BASE_URL=http://127.0.0.1:8004
```

## Predict

1. Create input file for inference at `data/processed/predict_input.csv`.
2. Run:

```bash
python scripts/predict.py
```

Output:
- `data/processed/predictions.csv`

## Tests

```bash
pytest
```

## Backend Service Planning

A high-level FastAPI backend architecture and delivery roadmap is documented in `docs/backend-service-plan.md`.

## Backend API

The repository now includes a first FastAPI backend slice with:

- `GET /v1/dashboard/bootstrap`
- `POST /v1/predictions`
- `GET /v1/predictions`
- `GET /v1/predictions/{prediction_id}`
- `GET /v1/predictions/{prediction_id}/trace`
- `GET /v1/predictions/{prediction_id}/events`
- `POST /v1/properties/normalize`
- `GET /v1/validation/scenarios`
- `POST /v1/validation/address-baseline`
- `POST /v1/validation/full-audit`
- `GET /v1/health`

Run locally with:

```bash
uvicorn house_price_prediction.api.main:app --reload
```

Run a backend smoke check without starting a separate server process:

```bash
python scripts/smoke_api.py
```

Run a fresh-db real-inference smoke check against the trained artifact and free fallback providers:

```bash
make smoke-real
```

Run migrations with:

```bash
alembic upgrade head
```

For local backend development without a trained artifact, `.env.example` enables `ENABLE_MOCK_PREDICTOR=true` so the API can return deterministic sample predictions while the orchestration and persistence layers are being built.

The backend now keeps normalized addresses and registered model versions in dedicated tables so repeated requests can share canonical address records and predictions can be traced to an explicit model registry entry.

Repeated requests for the same canonical address and active model version can now reuse the latest completed prediction instead of re-running enrichment and inference.

Reuse is controlled by `PREDICTION_REUSE_MAX_AGE_HOURS`, and provider selection is now configurable through `GEOCODING_PROVIDER` and `PROPERTY_DATA_PROVIDER`.

Provider execution policy is controlled by `PROVIDER_TIMEOUT_SECONDS` and `PROVIDER_MAX_RETRIES`. Upstream provider failures are mapped to `502 Bad Gateway` responses.

Feature completeness governance is controlled by `FEATURE_POLICY_NAME` (for example: `balanced-v1`, `quality-first-v1`, `land-first-v1`) so you can evolve feature weighting rules without changing route logic.

You can version and target policies with `FEATURE_POLICY_VERSION` and optional per-state overrides via `FEATURE_POLICY_STATE_OVERRIDES` (for example: `FL:quality-first-v1,TX:land-first-v1`).

API guardrails now reject PO box inputs, and the public free-provider path is explicitly limited to US addresses.

Property provider payloads now include `feature_source` and `feature_provenance` so downstream consumers can tell whether features came from Census context, heuristic backfill, or deterministic fake generation.

The prediction detail and recent-predictions contracts are now UI-facing read models: they include feature snapshot summaries and provider response summaries so a frontend can render trace, provenance, and prediction history without querying raw tables.

`GET /v1/dashboard/bootstrap` adds a compact UI bootstrap contract with runtime metadata, provider policy, recent predictions, and link templates so the frontend can hydrate its initial state from one stable endpoint.

For reused predictions, these read models also surface `source_prediction_id` and resolve provider/feature trace from the original prediction that produced the reused value.

For UI flows that need deeper lineage without loading the full detail payload everywhere, `GET /v1/predictions/{prediction_id}/trace` returns the provider trace, root lineage information, and the hop-by-hop reuse chain.

`GET /v1/predictions/{prediction_id}/events` returns ordered workflow events for the prediction request and supports `limit`, `offset`, and optional `event_name` filtering.

`GET /v1/validation/scenarios` now returns live-derived candidate addresses from recent traffic (not hardcoded fake scenario fixtures).

`POST /v1/validation/address-baseline` returns normalized location, pulled/missing feature coverage, and predicted value metadata for an address. Optional `expectations` can enforce pass/fail checks for completeness thresholds, required features, and feature bounds (including practical fields like bedrooms, total rooms, and living square footage).

`POST /v1/validation/full-audit` automates end-to-end live routing checks in one call: baseline + create prediction + detail + trace + events, and returns a consolidated issue list for production-readiness triage.

You can export the backend contract for frontend integration with:

```bash
make export-openapi
```

This writes the current OpenAPI schema to `docs/openapi.json`.

`make ci-verify` runs both smoke flows against isolated fresh SQLite files so local reruns do not hide behavior behind prediction reuse.

The default no-key setup now uses free-provider fallbacks:

- `GEOCODING_PROVIDER=free-fallback` tries Nominatim, then US Census geocoding, then local fake normalization
- `PROPERTY_DATA_PROVIDER=free-fallback` tries tract-level US Census housing context, then heuristic address-derived features, then deterministic fake features

## Optional Make Commands

```bash
make setup
make bootstrap-data
make bootstrap-data-snapshot
make refresh-live-pipeline
make train
make predict
make test
make smoke-api
make smoke-real
make ci-verify
make export-openapi
```
