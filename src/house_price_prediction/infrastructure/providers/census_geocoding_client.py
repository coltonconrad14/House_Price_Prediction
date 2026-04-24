from __future__ import annotations

import httpx

from house_price_prediction.domain.contracts.prediction_contracts import (
    AddressPayload,
    GeocodingResultContract,
    NormalizedAddress,
    ProviderResponseContract,
)
from datetime import UTC, datetime
from house_price_prediction.infrastructure.providers.resilient import NonRetryableProviderError


class CensusGeocodingClient:
    def __init__(self, base_url: str = "https://geocoding.geo.census.gov/geocoder") -> None:
        self._base_url = base_url.rstrip("/")

    def normalize(self, address_payload: AddressPayload) -> GeocodingResultContract:
        query = ", ".join(
            [
                address_payload.address_line_1.strip(),
                address_payload.city.strip(),
                address_payload.state.strip(),
                address_payload.postal_code.strip(),
                address_payload.country.strip(),
            ]
        )
        response = httpx.get(
            f"{self._base_url}/locations/onelineaddress",
            params={
                "address": query,
                "benchmark": "Public_AR_Current",
                "format": "json",
            },
            headers={"User-Agent": "house-price-prediction-backend/0.1"},
        )
        response.raise_for_status()
        matches = response.json().get("result", {}).get("addressMatches", [])
        if not matches:
            raise NonRetryableProviderError("Census geocoder returned no address matches.")

        best_match = matches[0]
        coordinates = best_match.get("coordinates", {})

        normalized_line_1 = " ".join(address_payload.address_line_1.strip().upper().split())
        normalized_city = " ".join(address_payload.city.strip().upper().split())
        normalized_state = address_payload.state.strip().upper()
        normalized_postal_code = address_payload.postal_code.strip().upper()
        normalized_country = address_payload.country.strip().upper()
        normalized_line_2 = (
            " ".join(address_payload.address_line_2.strip().upper().split())
            if address_payload.address_line_2
            else None
        )

        normalized_address = NormalizedAddress(
            address_line_1=normalized_line_1,
            address_line_2=normalized_line_2,
            city=normalized_city,
            state=normalized_state,
            postal_code=normalized_postal_code,
            country=normalized_country,
            formatted_address=(
                f"{normalized_line_1}, {normalized_city}, {normalized_state} "
                f"{normalized_postal_code}, {normalized_country}"
            ),
            latitude=float(coordinates.get("y")) if coordinates.get("y") is not None else None,
            longitude=float(coordinates.get("x")) if coordinates.get("x") is not None else None,
            geocoding_source="census",
        )
        return GeocodingResultContract(
            normalized_address=normalized_address,
            provider_response=ProviderResponseContract(
                provider_name="census_geocoding",
                status="success",
                payload={"result": best_match},
                fetched_at=datetime.now(UTC),
            ),
        )