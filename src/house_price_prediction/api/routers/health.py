from __future__ import annotations

from fastapi import APIRouter, Depends

from house_price_prediction.api.dependencies import get_prediction_runtime, get_settings
from house_price_prediction.config import Settings
from house_price_prediction.domain.contracts.prediction_contracts import HealthResponse
from house_price_prediction.infrastructure.model_runtime.predictor import PredictionRuntime

router = APIRouter(tags=["health"])


@router.get("/v1/health", response_model=HealthResponse)
def get_health(
    settings: Settings = Depends(get_settings),
    prediction_runtime: PredictionRuntime = Depends(get_prediction_runtime),
) -> HealthResponse:
    live_mode_issues: list[str] = []
    if settings.enable_mock_predictor:
        live_mode_issues.append("Mock predictor is enabled; set ENABLE_MOCK_PREDICTOR=false.")
    if settings.geocoding_provider.strip().lower() == "fake":
        live_mode_issues.append("Geocoding provider is fake; use a live geocoder provider.")
    if settings.property_data_provider.strip().lower() == "fake":
        live_mode_issues.append("Property data provider is fake; use a live property provider.")
    if not prediction_runtime.is_available():
        live_mode_issues.append("Model artifact is unavailable for inference.")

    return HealthResponse(
        status="ok",
        environment=settings.app_env,
        app_name=settings.app_name,
        model_name=settings.model_name,
        model_version=settings.model_version,
        model_available=prediction_runtime.is_available(),
        mock_predictor_enabled=settings.enable_mock_predictor,
        property_data_provider=settings.property_data_provider,
        geocoding_provider=settings.geocoding_provider,
        provider_timeout_seconds=settings.provider_timeout_seconds,
        provider_max_retries=settings.provider_max_retries,
        prediction_reuse_max_age_hours=settings.prediction_reuse_max_age_hours,
        provider_response_cache_max_age_hours=settings.provider_response_cache_max_age_hours,
        feature_policy_name=settings.feature_policy_name,
        feature_policy_version=settings.feature_policy_version,
        feature_policy_state_override_count=len(settings.feature_policy_state_overrides or {}),
        walkscore_enrichment_active=bool(settings.walkscore_api_key),
        live_mode_ready=len(live_mode_issues) == 0,
        live_mode_issues=live_mode_issues,
    )