from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressPayload,
    GeocodingResultContract,
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.base import (
    GeocodingProvider,
    PropertyDataProvider,
)
from house_price_prediction.telemetry import get_logger

logger = get_logger(__name__)


class ProviderExecutionError(RuntimeError):
    def __init__(self, provider_name: str, message: str) -> None:
        self.provider_name = provider_name
        super().__init__(message)


class NonRetryableProviderError(RuntimeError):
    """Raised by providers when the failure is deterministic (e.g. address not found).

    The resilient wrapper will not retry on this error — it immediately wraps it
    in ProviderExecutionError and surfaces it to the caller.
    """


@dataclass(frozen=True)
class ResilientPropertyDataProvider:
    provider_name: str
    delegate: PropertyDataProvider
    timeout_seconds: float
    max_retries: int

    def fetch_property_features(
        self,
        normalized_address: NormalizedAddress,
    ) -> ProviderResponseContract:
        attempts = self.max_retries + 1
        last_error: Exception | None = None

        for _ in range(attempts):
            try:
                logger.info(
                    "provider_call_start provider=%s operation=property_features timeout=%s retries=%s",
                    self.provider_name,
                    self.timeout_seconds,
                    self.max_retries,
                )
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.delegate.fetch_property_features, normalized_address)
                    result = future.result(timeout=self.timeout_seconds)
                logger.info(
                    "provider_call_success provider=%s operation=property_features",
                    self.provider_name,
                )
                return result
            except FutureTimeoutError:
                logger.warning(
                    "provider_call_timeout provider=%s operation=property_features timeout=%s",
                    self.provider_name,
                    self.timeout_seconds,
                )
                last_error = ProviderExecutionError(
                    self.provider_name,
                    f"Provider '{self.provider_name}' timed out after {self.timeout_seconds:.1f}s.",
                )
            except NonRetryableProviderError as exc:
                logger.warning(
                    "provider_call_no_match provider=%s operation=property_features error=%s",
                    self.provider_name,
                    exc,
                )
                raise ProviderExecutionError(
                    self.provider_name,
                    f"Provider '{self.provider_name}' failed: {exc}",
                ) from exc
            except Exception as exc:
                logger.warning(
                    "provider_call_failure provider=%s operation=property_features error=%s",
                    self.provider_name,
                    exc,
                )
                last_error = ProviderExecutionError(
                    self.provider_name,
                    f"Provider '{self.provider_name}' failed: {exc}",
                )

        assert last_error is not None
        raise last_error


@dataclass(frozen=True)
class ResilientGeocodingProvider:
    provider_name: str
    delegate: GeocodingProvider
    timeout_seconds: float
    max_retries: int

    def normalize(self, address_payload: AddressPayload) -> GeocodingResultContract:
        attempts = self.max_retries + 1
        last_error: Exception | None = None

        for _ in range(attempts):
            try:
                logger.info(
                    "provider_call_start provider=%s operation=geocoding timeout=%s retries=%s",
                    self.provider_name,
                    self.timeout_seconds,
                    self.max_retries,
                )
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.delegate.normalize, address_payload)
                    result = future.result(timeout=self.timeout_seconds)
                logger.info(
                    "provider_call_success provider=%s operation=geocoding",
                    self.provider_name,
                )
                return result
            except FutureTimeoutError:
                logger.warning(
                    "provider_call_timeout provider=%s operation=geocoding timeout=%s",
                    self.provider_name,
                    self.timeout_seconds,
                )
                last_error = ProviderExecutionError(
                    self.provider_name,
                    f"Provider '{self.provider_name}' timed out after {self.timeout_seconds:.1f}s.",
                )
            except NonRetryableProviderError as exc:
                logger.warning(
                    "provider_call_no_match provider=%s operation=geocoding error=%s",
                    self.provider_name,
                    exc,
                )
                raise ProviderExecutionError(
                    self.provider_name,
                    f"Provider '{self.provider_name}' failed: {exc}",
                ) from exc
            except Exception as exc:
                logger.warning(
                    "provider_call_failure provider=%s operation=geocoding error=%s",
                    self.provider_name,
                    exc,
                )
                last_error = ProviderExecutionError(
                    self.provider_name,
                    f"Provider '{self.provider_name}' failed: {exc}",
                )

        assert last_error is not None
        raise last_error