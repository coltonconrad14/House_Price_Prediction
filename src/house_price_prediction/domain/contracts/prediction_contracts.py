from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AddressPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    address_line_1: str = Field(min_length=1, max_length=120)
    address_line_2: str | None = Field(default=None, max_length=120)
    city: str = Field(min_length=1, max_length=80)
    state: str = Field(min_length=2, max_length=30)
    postal_code: str = Field(min_length=3, max_length=20)
    country: str = Field(default="US", min_length=2, max_length=60)


class PredictionRequestPayload(AddressPayload):
    requested_by: str | None = Field(default=None, max_length=120)


class NormalizedAddress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    address_line_1: str
    address_line_2: str | None = None
    city: str
    state: str
    postal_code: str
    country: str
    formatted_address: str
    latitude: float | None = None
    longitude: float | None = None
    geocoding_source: str | None = None


class FeatureVectorContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    model_name: str
    model_version: str
    features: dict[str, Any]
    completeness_score: float
    feature_policy_name: str | None = None
    feature_policy_version: str | None = None
    weight_total: float | None = None


class ProviderResponseContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    status: Literal["success", "failed"]
    payload: dict[str, Any]
    fetched_at: datetime


class ProviderResponseSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    status: Literal["success", "failed"]
    fetched_at: datetime
    feature_source: str | None = None
    feature_provenance: dict[str, Any] | None = None
    payload_preview: dict[str, Any]


class FeatureSnapshotSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_snapshot_id: UUID
    model_name: str
    model_version: str
    completeness_score: float
    feature_count: int
    populated_feature_count: int
    features: dict[str, Any]


class GeocodingResultContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_address: NormalizedAddress
    provider_response: ProviderResponseContract


PredictionStatus = Literal["received", "completed", "failed"]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    prediction_id: UUID
    correlation_id: UUID
    status: PredictionStatus
    predicted_price: float
    currency: str = "USD"
    confidence_score: float | None = None
    model_name: str
    model_version: str
    feature_snapshot_id: UUID
    normalized_address: NormalizedAddress
    submitted_at: datetime
    generated_at: datetime
    was_reused: bool = False
    source_prediction_id: UUID | None = None
    selected_feature_policy_name: str | None = None
    selected_feature_policy_version: str | None = None


class PredictionDetailResponse(PredictionResponse):
    feature_snapshot: FeatureSnapshotSummary
    provider_responses: list[ProviderResponseSummary]
    error_message: str | None = None


class PredictionTraceNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    prediction_id: UUID
    generated_at: datetime
    was_reused: bool
    source_prediction_id: UUID | None = None


class WorkflowEventItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_name: str
    payload: dict[str, Any]
    occurred_at: datetime


class PredictionTraceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    prediction_id: UUID
    source_prediction_id: UUID | None = None
    root_prediction_id: UUID
    was_reused: bool
    normalized_address: NormalizedAddress
    feature_snapshot: FeatureSnapshotSummary
    provider_responses: list[ProviderResponseSummary]
    trace_nodes: list[PredictionTraceNode]
    workflow_events: list[WorkflowEventItem] = Field(default_factory=list)


class PredictionListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    prediction_id: UUID
    status: PredictionStatus
    predicted_price: float
    currency: str = "USD"
    model_name: str
    model_version: str
    normalized_address: NormalizedAddress
    submitted_at: datetime
    generated_at: datetime
    was_reused: bool = False
    source_prediction_id: UUID | None = None
    requested_by: str | None = None
    feature_source: str | None = None
    selected_feature_policy_name: str | None = None
    selected_feature_policy_version: str | None = None


class PredictionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PredictionListItem]
    total: int = 0
    limit: int = 10
    offset: int = 0


class FeaturePolicyDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str
    emphasis_features: list[str]


class FeaturePolicyCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_policy_name: str
    default_policy_version: str
    state_overrides: dict[str, str]
    policies: list[FeaturePolicyDescriptor]


class FeaturePolicySimulationRequest(AddressPayload):
    policy_names: list[str] | None = None


class FeaturePolicySimulationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_name: str
    policy_version: str
    predicted_price: float
    completeness_score: float
    weight_total: float | None = None


class FeaturePolicySimulationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_address: NormalizedAddress
    provider_name: str
    simulations: list[FeaturePolicySimulationItem]


class FeatureBoundExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum: float
    maximum: float


class BaselineScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    label: str
    description: str
    payload: AddressPayload
    min_completeness_score: float = 0.95
    required_features: list[str] = Field(default_factory=list)
    feature_bounds: dict[str, FeatureBoundExpectation] = Field(default_factory=dict)


class BaselineScenarioCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenarios: list[BaselineScenario]


class BaselineLocationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_address: NormalizedAddress
    geocoding_provider: str
    geocoding_payload_preview: dict[str, Any]


class BaselineFeatureObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_features: list[str]
    pulled_features: list[str]
    missing_features: list[str]
    pulled_feature_count: int
    missing_feature_count: int
    completeness_score: float
    weight_total: float | None = None
    selected_feature_policy_name: str | None = None
    selected_feature_policy_version: str | None = None
    feature_values: dict[str, Any] = Field(default_factory=dict)
    key_feature_values: dict[str, Any] = Field(default_factory=dict)


class BaselineCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_name: str
    status: Literal["passed", "failed", "skipped"]
    message: str


class BaselineAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_status: Literal["passed", "failed", "not_evaluated"]
    checks: list[BaselineCheckResult]


class BaselineExpectationsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_completeness_score: float | None = None
    required_features: list[str] = Field(default_factory=list)
    feature_bounds: dict[str, FeatureBoundExpectation] = Field(default_factory=dict)


class AddressBaselineRequest(AddressPayload):
    expectations: BaselineExpectationsInput | None = None


class BaselineValueObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_price: float
    model_name: str
    model_version: str
    property_provider: str
    property_feature_source: str | None = None
    property_feature_provenance: dict[str, Any] | None = None


class AddressBaselineResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str
    evaluated_at: datetime
    input_address: AddressPayload
    location: BaselineLocationObservation
    features: BaselineFeatureObservation
    value: BaselineValueObservation
    assessment: BaselineAssessment


class FullAuditRequest(PredictionRequestPayload):
    expectations: BaselineExpectationsInput | None = None


class FullAuditResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str
    audited_at: datetime
    baseline: AddressBaselineResponse
    prediction: PredictionResponse
    prediction_detail: PredictionDetailResponse | None = None
    prediction_trace: PredictionTraceResponse | None = None
    prediction_events: PredictionWorkflowEventsResponse | None = None
    issues: list[str]


class ScenarioPipelineResult(BaseModel):
    """Result of running one registered scenario through the full audit pipeline."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    label: str
    category: str
    pipeline_status: Literal["pass", "fail", "error"]
    issues: list[str]
    completeness_score: float | None = None
    predicted_price: float | None = None
    key_feature_values: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class ScenarioBatchPipelineRequest(BaseModel):
    """Request to run a subset (or all) registered scenarios through the pipeline."""

    model_config = ConfigDict(extra="forbid")

    scenario_ids: list[str] | None = None


class ScenarioBatchPipelineResponse(BaseModel):
    """Aggregated result of a scenario batch pipeline run."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str
    started_at: datetime
    completed_at: datetime
    total: int
    passed: int
    failed: int
    errors: int
    results: list[ScenarioPipelineResult]


class PredictionWorkflowEventsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    prediction_id: UUID
    total_count: int
    limit: int
    offset: int
    event_name: str | None = None
    sort: Literal["asc", "desc"] = "asc"
    events: list[WorkflowEventItem]


class DashboardEventSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: UUID
    total_events: int
    latest_event_name: str | None = None
    latest_event_at: datetime | None = None
    recent_event_names: list[str]


class RuntimeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_name: str
    environment: str
    model_name: str
    model_version: str
    model_available: bool
    mock_predictor_enabled: bool


class ProviderPolicySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_data_provider: str
    geocoding_provider: str
    provider_timeout_seconds: float
    provider_max_retries: int
    prediction_reuse_max_age_hours: int
    feature_policy_name: str
    feature_policy_version: str
    feature_policy_state_overrides: dict[str, str]


class UiLinkTemplates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions_list: str
    prediction_detail: str
    prediction_trace: str
    prediction_create: str
    address_normalize: str


class DashboardBootstrapResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    contract_version: str
    runtime: RuntimeSummary
    provider_policy: ProviderPolicySummary
    recent_predictions: list[PredictionListItem]
    event_summary: DashboardEventSummary | None = None
    links: UiLinkTemplates


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    environment: str
    app_name: str
    model_name: str
    model_version: str
    model_available: bool
    mock_predictor_enabled: bool
    property_data_provider: str
    geocoding_provider: str
    provider_timeout_seconds: float
    provider_max_retries: int
    prediction_reuse_max_age_hours: int
    provider_response_cache_max_age_hours: int
    feature_policy_name: str
    feature_policy_version: str
    feature_policy_state_override_count: int
    walkscore_enrichment_active: bool = False
    live_mode_ready: bool
    live_mode_issues: list[str] = Field(default_factory=list)


class ApiEndpointDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    method: str
    path: str
    purpose: str


class ApiExamplePayloads(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_request: AddressPayload
    normalization_request: AddressPayload
    baseline_request: AddressPayload


class ApiCapabilitiesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str
    generated_at: datetime
    runtime: RuntimeSummary
    provider_policy: ProviderPolicySummary
    model_expected_features: list[str]
    live_mode_ready: bool
    live_mode_issues: list[str] = Field(default_factory=list)
    endpoints: list[ApiEndpointDescriptor]
    examples: ApiExamplePayloads


class LiveFeatureCandidateItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: UUID
    request_id: UUID
    submitted_at: datetime
    generated_at: datetime
    predicted_price: float
    completeness_score: float
    was_reused: bool
    model_name: str
    model_version: str
    selected_feature_policy_name: str | None = None
    selected_feature_policy_version: str | None = None
    feature_source: str | None = None
    provider_name: str | None = None
    normalized_address: NormalizedAddress
    features: dict[str, Any]


class LiveFeatureCandidatesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str
    generated_at: datetime
    total: int
    limit: int
    offset: int
    min_completeness_score: float
    include_reused: bool
    items: list[LiveFeatureCandidateItem]