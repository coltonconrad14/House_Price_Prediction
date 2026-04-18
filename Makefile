.PHONY: setup bootstrap-data train predict test lint clean smoke-api smoke-real ci-verify export-openapi run-api run-api-live run-dashboard live-address-audit export-live-features refresh-live-pipeline bootstrap-data-snapshot

BOOTSTRAP_BASE_URL ?= http://127.0.0.1:8000

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements-dev.txt
	. .venv/bin/activate && pip install -e .

train:
	python scripts/train.py --min-rows=25

bootstrap-data:
	python scripts/bootstrap_training_data.py --base-url=$(BOOTSTRAP_BASE_URL) --output=data/processed/live_feature_store.jsonl --metadata-output=data/processed/live_feature_store.jsonl.meta.json --min-completeness-score=0.9 --max-rows=5000 --min-output-rows=1

bootstrap-data-snapshot:
	python scripts/bootstrap_training_data.py --base-url=$(BOOTSTRAP_BASE_URL) --output=data/processed/live_feature_store.jsonl --metadata-output=data/processed/live_feature_store.jsonl.meta.json --snapshot-dir=data/processed/snapshots --snapshot-prefix=live_feature_store --min-completeness-score=0.9 --max-rows=5000 --min-output-rows=25

refresh-live-pipeline:
	python scripts/bootstrap_training_data.py --base-url=$(BOOTSTRAP_BASE_URL) --output=data/processed/live_feature_store.jsonl --metadata-output=data/processed/live_feature_store.jsonl.meta.json --snapshot-dir=data/processed/snapshots --snapshot-prefix=live_feature_store --min-completeness-score=0.9 --max-rows=5000 --min-output-rows=25
	python scripts/train.py --min-rows=25

predict:
	python scripts/predict.py

test:
	pytest

smoke-api:
	python scripts/smoke_api.py

smoke-real:
	rm -f data/processed/real_validation.db
	ENABLE_MOCK_PREDICTOR=false GEOCODING_PROVIDER=free-fallback PROPERTY_DATA_PROVIDER=free-fallback PREDICTION_REUSE_MAX_AGE_HOURS=0 DATABASE_URL=sqlite:///data/processed/real_validation.db python scripts/smoke_api.py

run-api:
	APP_ENV=test DATABASE_URL=sqlite:///data/processed/live_validation.db uvicorn house_price_prediction.api.main:app --host 0.0.0.0 --port 8000

run-api-live:
	APP_ENV=test ENABLE_MOCK_PREDICTOR=false GEOCODING_PROVIDER=free-fallback PROPERTY_DATA_PROVIDER=free-fallback PREDICTION_REUSE_MAX_AGE_HOURS=0 DATABASE_URL=sqlite:///data/processed/real_validation.db uvicorn house_price_prediction.api.main:app --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --server.enableCORS false --server.enableXsrfProtection false

live-address-audit:
	ENABLE_MOCK_PREDICTOR=false GEOCODING_PROVIDER=free-fallback PROPERTY_DATA_PROVIDER=free-fallback python scripts/live_address_audit.py --base-url=http://127.0.0.1:8000 --address-line-1="1600 Pennsylvania Ave NW" --city="Washington" --state="DC" --postal-code="20500" --country="US"

export-live-features:
	python scripts/export_live_feature_candidates.py --base-url=http://127.0.0.1:8000 --output=data/processed/live_feature_candidates.csv --min-completeness-score=0.8 --max-rows=5000

export-openapi:
	python scripts/export_openapi.py

ci-verify:
	python scripts/bootstrap_training_data.py
	python scripts/train.py --min-rows=2
	pytest
	rm -f data/processed/smoke_validation.db
	DATABASE_URL=sqlite:///data/processed/smoke_validation.db PREDICTION_REUSE_MAX_AGE_HOURS=0 python scripts/smoke_api.py
	rm -f data/processed/real_validation.db
	ENABLE_MOCK_PREDICTOR=false GEOCODING_PROVIDER=free-fallback PROPERTY_DATA_PROVIDER=free-fallback PREDICTION_REUSE_MAX_AGE_HOURS=0 DATABASE_URL=sqlite:///data/processed/real_validation.db python scripts/smoke_api.py
	python scripts/export_openapi.py

lint:
	python -m py_compile src/house_price_prediction/*.py scripts/*.py

clean:
	rm -rf .pytest_cache
