from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    raw_data_path: Path
    target_column: str
    model_path: Path
    test_size: float
    random_state: int
    app_name: str
    app_env: str
    api_host: str
    api_port: int
    database_url: str
    model_name: str
    model_version: str
    enable_mock_predictor: bool
    property_data_provider: str
    geocoding_provider: str
    prediction_reuse_max_age_hours: int
    provider_timeout_seconds: float
    provider_max_retries: int
    provider_response_cache_max_age_hours: int = 24
    training_min_rows: int = 0
    feature_policy_name: str = "balanced-v1"
    feature_policy_version: str = "v1"
    feature_policy_state_overrides: dict[str, str] = field(default_factory=dict)
    walkscore_api_key: str = ""


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_feature_policy_state_overrides(raw: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not raw.strip():
        return overrides

    for chunk in raw.split(","):
        entry = chunk.strip()
        if not entry or ":" not in entry:
            continue
        state, policy_name = entry.split(":", 1)
        normalized_state = state.strip().upper()
        normalized_policy_name = policy_name.strip()
        if normalized_state and normalized_policy_name:
            overrides[normalized_state] = normalized_policy_name
    return overrides


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load settings from environment variables with sensible defaults."""
    load_dotenv()

    return Settings(
        raw_data_path=Path(os.getenv("RAW_DATA_PATH", "data/processed/live_feature_store.jsonl")),
        target_column=os.getenv("TARGET_COLUMN", "SalePrice"),
        model_path=Path(os.getenv("MODEL_PATH", "models/house_price_model.joblib")),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        app_name=os.getenv("APP_NAME", "House Price Prediction API"),
        app_env=os.getenv("APP_ENV", "development"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        database_url=os.getenv(
            "DATABASE_URL", "sqlite:///data/processed/house_price_prediction.db"
        ),
        model_name=os.getenv("MODEL_NAME", "house-price-random-forest"),
        model_version=os.getenv("MODEL_VERSION", "0.1.0"),
        enable_mock_predictor=_get_bool_env("ENABLE_MOCK_PREDICTOR", True),
        property_data_provider=os.getenv("PROPERTY_DATA_PROVIDER", "fake"),
        geocoding_provider=os.getenv("GEOCODING_PROVIDER", "fake"),
        prediction_reuse_max_age_hours=int(os.getenv("PREDICTION_REUSE_MAX_AGE_HOURS", "24")),
        provider_response_cache_max_age_hours=int(
            os.getenv("PROVIDER_RESPONSE_CACHE_MAX_AGE_HOURS", "24")
        ),
        training_min_rows=int(os.getenv("TRAINING_MIN_ROWS", "0")),
        provider_timeout_seconds=float(os.getenv("PROVIDER_TIMEOUT_SECONDS", "3.0")),
        provider_max_retries=int(os.getenv("PROVIDER_MAX_RETRIES", "2")),
        feature_policy_name=os.getenv("FEATURE_POLICY_NAME", "balanced-v1"),
        feature_policy_version=os.getenv("FEATURE_POLICY_VERSION", "v1"),
        feature_policy_state_overrides=_parse_feature_policy_state_overrides(
            os.getenv("FEATURE_POLICY_STATE_OVERRIDES", "")
        ),
        walkscore_api_key=os.getenv("WALKSCORE_API_KEY", ""),
    )
