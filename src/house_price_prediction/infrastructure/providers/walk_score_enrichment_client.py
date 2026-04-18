from __future__ import annotations

from datetime import UTC, datetime

import httpx

from house_price_prediction.domain.contracts.prediction_contracts import (
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.base import PropertyDataProvider

_WALK_SCORE_URL = "https://api.walkscore.com/score"

# Walk / transit score thresholds used for feature adjustment
_WALK_URBAN = 70       # >= → 2Story / dense lot
_WALK_VERY_URBAN = 85  # >= → even smaller lot
_TRANSIT_HIGH = 60     # >= → 0 garage cars (excellent transit)
_TRANSIT_MED = 35      # >= → 1 garage car


class WalkScoreEnrichmentClient:
    """Wraps any PropertyDataProvider and enriches its output with Walk Score signals.

    When ``api_key`` is empty or the API call fails, the base provider's
    response is returned unchanged — this client never degrades the pipeline.

    Walk / transit / bike scores are used to override Census-derived estimates
    for ``GarageCars``, ``GarageArea``, ``HouseStyle``, and ``LotArea`` with
    location-specific walkability data, which is a strong correlate of home
    prices and neighbourhood density.

    Enrichment signals are stored in the payload under ``walkscore_*`` keys for
    observability and downstream feature policy analysis.
    """

    def __init__(
        self,
        base_provider: PropertyDataProvider,
        api_key: str = "",
    ) -> None:
        self._base = base_provider
        self._api_key = (api_key or "").strip()

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def fetch_property_features(
        self,
        normalized_address: NormalizedAddress,
    ) -> ProviderResponseContract:
        base_response = self._base.fetch_property_features(normalized_address)

        if not self._api_key:
            return base_response
        if normalized_address.latitude is None or normalized_address.longitude is None:
            return base_response

        try:
            walk_data = self._fetch_walk_scores(normalized_address)
            if walk_data is None:
                return base_response

            enriched = self._apply_walk_signals(dict(base_response.payload), walk_data)
            enriched["feature_provenance"] = self._build_provenance(
                base_response.payload.get("feature_provenance"),
                walk_data,
            )
            return ProviderResponseContract(
                provider_name=f"{base_response.provider_name}+walkscore",
                status="success",
                payload=enriched,
                fetched_at=datetime.now(UTC),
            )
        except Exception:
            # Walk Score is supplemental — never fail the pipeline on its behalf
            return base_response

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _fetch_walk_scores(self, address: NormalizedAddress) -> dict | None:
        """Call the Walk Score API and return a normalised signal dict."""
        response = httpx.get(
            _WALK_SCORE_URL,
            params={
                "format": "json",
                "address": address.formatted_address,
                "lat": address.latitude,
                "lon": address.longitude,
                "transit": 1,
                "bike": 1,
                "wsapikey": self._api_key,
            },
            headers={"User-Agent": "house-price-prediction-backend/0.1"},
            timeout=5.0,
        )
        response.raise_for_status()
        data = response.json()

        # status == 1 means a valid score was returned
        if data.get("status") != 1:
            return None

        return {
            "walk_score": data.get("walkscore"),
            "walk_description": data.get("description"),
            "transit_score": (data.get("transit") or {}).get("score"),
            "transit_description": (data.get("transit") or {}).get("description"),
            "bike_score": (data.get("bike") or {}).get("score"),
            "bike_description": (data.get("bike") or {}).get("description"),
        }

    @staticmethod
    def _apply_walk_signals(features: dict, walk_data: dict) -> dict:
        """Override feature estimates with Walk Score-calibrated values."""
        walk = walk_data.get("walk_score") or 0
        transit = walk_data.get("transit_score") or 0

        # GarageCars / GarageArea: excellent transit reduces parking demand
        if transit >= _TRANSIT_HIGH:
            garage_cars = 0
        elif transit >= _TRANSIT_MED:
            garage_cars = 1
        else:
            garage_cars = features.get("GarageCars", 2)
        features["GarageCars"] = garage_cars
        features["GarageArea"] = garage_cars * 260

        # HouseStyle: walkable / urban areas skew toward multi-story forms
        features["HouseStyle"] = "2Story" if walk >= _WALK_URBAN else "1Story"

        # LotArea: denser walkable areas have smaller lots
        base_lot = features.get("LotArea", 7000)
        if walk >= _WALK_VERY_URBAN:
            features["LotArea"] = max(int(base_lot * 0.45), 1500)
        elif walk >= _WALK_URBAN:
            features["LotArea"] = max(int(base_lot * 0.65), 2500)
        # Scores < 70 leave LotArea unchanged from Census estimate

        # Store Walk Score signals for observability
        features["walkscore_walk"] = walk_data.get("walk_score")
        features["walkscore_transit"] = walk_data.get("transit_score")
        features["walkscore_bike"] = walk_data.get("bike_score")
        features["walkscore_walk_description"] = walk_data.get("walk_description")

        return features

    @staticmethod
    def _build_provenance(base: dict | None, walk_data: dict) -> dict:
        provenance: dict = dict(base) if base else {}
        providers: list = list(provenance.get("providers", []))
        if "walkscore" not in providers:
            providers.append("walkscore")
        provenance["providers"] = providers
        provenance["walkscore_signals"] = {
            "walk_score": walk_data.get("walk_score"),
            "walk_description": walk_data.get("walk_description"),
            "transit_score": walk_data.get("transit_score"),
            "transit_description": walk_data.get("transit_description"),
            "bike_score": walk_data.get("bike_score"),
            "bike_description": walk_data.get("bike_description"),
        }
        return provenance
