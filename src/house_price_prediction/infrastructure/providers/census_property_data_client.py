from __future__ import annotations

from datetime import UTC, datetime

import httpx

from house_price_prediction.domain.contracts.prediction_contracts import (
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.base import PropertyDataProvider
from house_price_prediction.infrastructure.providers.property_type_classifier import (
    classify_property_type,
)
from house_price_prediction.infrastructure.providers.resilient import NonRetryableProviderError


class CensusPropertyDataClient:
    def __init__(
        self,
        fallback_provider: PropertyDataProvider | None = None,
        geocoder_base_url: str = "https://geocoding.geo.census.gov/geocoder",
        census_api_base_url: str = "https://api.census.gov/data/2022/acs/acs5",
    ) -> None:
        self._fallback_provider = fallback_provider
        self._geocoder_base_url = geocoder_base_url.rstrip("/")
        self._census_api_base_url = census_api_base_url.rstrip("/")

    def fetch_property_features(
        self,
        normalized_address: NormalizedAddress,
    ) -> ProviderResponseContract:
        fallback_response = (
            self._fallback_provider.fetch_property_features(normalized_address)
            if self._fallback_provider is not None
            else None
        )

        try:
            if normalized_address.latitude is None or normalized_address.longitude is None:
                raise NonRetryableProviderError("Coordinates are required for census context enrichment.")

            geography = self._lookup_census_tract(
                latitude=normalized_address.latitude,
                longitude=normalized_address.longitude,
            )
            census_context = self._fetch_census_context(geography)
            derived_features = self._derive_features(census_context, geography)

            payload = dict(fallback_response.payload) if fallback_response else {}
            payload.update(derived_features)
            payload["feature_source"] = "census_context"
            payload["feature_provenance"] = self._build_feature_provenance(
                fallback_response=fallback_response,
                geography=geography,
                used_census=True,
            )

            return ProviderResponseContract(
                provider_name=("census_context_with_backfill" if fallback_response is not None else "census_context"),
                status="success",
                payload=payload,
                fetched_at=datetime.now(UTC),
            )
        except Exception:
            if fallback_response is not None:
                payload = dict(fallback_response.payload)
                payload["feature_provenance"] = self._build_feature_provenance(
                    fallback_response=fallback_response,
                    geography=None,
                    used_census=False,
                )
                return ProviderResponseContract(
                    provider_name=f"{fallback_response.provider_name}_fallback",
                    status="success",
                    payload=payload,
                    fetched_at=datetime.now(UTC),
                )
            raise

    def _lookup_census_tract(self, latitude: float, longitude: float) -> dict[str, str]:
        response = httpx.get(
            f"{self._geocoder_base_url}/geographies/coordinates",
            params={
                "x": longitude,
                "y": latitude,
                "benchmark": "Public_AR_Current",
                "vintage": "Current_Current",
                "layers": "Census Tracts",
                "format": "json",
            },
            headers={"User-Agent": "house-price-prediction-backend/0.1"},
        )
        response.raise_for_status()
        geographies = response.json().get("result", {}).get("geographies", {})
        tract_rows = geographies.get("Census Tracts") or geographies.get("2020 Census Tracts") or []
        if not tract_rows:
            raise RuntimeError("Census geography lookup returned no tract information.")
        tract = tract_rows[0]
        return {
            "state": str(tract.get("STATE") or tract.get("STATEFP")),
            "county": str(tract.get("COUNTY") or tract.get("COUNTYFP")),
            "tract": str(tract.get("TRACT") or tract.get("TRACTCE")),
            "name": str(tract.get("NAME") or tract.get("BASENAME") or "Unknown Tract"),
        }

    def _fetch_census_context(self, geography: dict[str, str]) -> dict[str, str]:
        # Core housing variables
        #   B25077_001E  median home value
        #   B25035_001E  median year structure built
        #   B25018_001E  median number of rooms
        # Extended neighborhood-quality variables
        #   B19013_001E  median household income
        #   B25064_001E  median gross rent
        #   B25003_001E  total housing units (all tenures)
        #   B25003_002E  owner-occupied housing units
        #   B25071_001E  gross rent as % of household income (rent-burden)
        #   B01003_001E  total population
        acs_fields = (
            "NAME"
            ",B25077_001E,B25035_001E,B25018_001E"
            ",B19013_001E,B25064_001E"
            ",B25003_001E,B25003_002E"
            ",B25071_001E,B01003_001E"
        )
        response = httpx.get(
            self._census_api_base_url,
            params={
                "get": acs_fields,
                "for": f"tract:{geography['tract']}",
                "in": f"state:{geography['state']} county:{geography['county']}",
            },
            headers={"User-Agent": "house-price-prediction-backend/0.1"},
        )
        response.raise_for_status()
        rows = response.json()
        if len(rows) < 2:
            raise RuntimeError("ACS returned no tract-level housing context.")
        headers = rows[0]
        values = rows[1]
        return dict(zip(headers, values, strict=True))

    def _derive_features(
        self,
        census_context: dict[str, str],
        geography: dict[str, str],
    ) -> dict[str, int | float | str | None]:
        # Core housing signals
        median_home_value = self._safe_int(census_context.get("B25077_001E"))
        median_year_built = self._safe_int(census_context.get("B25035_001E"))
        median_rooms = self._safe_float(census_context.get("B25018_001E"))

        # Extended neighborhood signals
        median_income = self._safe_int(census_context.get("B19013_001E"))
        median_rent = self._safe_int(census_context.get("B25064_001E"))
        total_units = self._safe_int(census_context.get("B25003_001E"))
        owner_units = self._safe_int(census_context.get("B25003_002E"))
        rent_burden_pct = self._safe_float(census_context.get("B25071_001E"))
        tract_population = self._safe_int(census_context.get("B01003_001E"))

        # Derived signals
        owner_rate: float | None = (
            owner_units / total_units if total_units and owner_units else None
        )
        # Persons-per-unit proxy for density (US avg ~2.5; urban < 2.0; suburban > 3.0)
        persons_per_unit: float | None = (
            tract_population / total_units if tract_population and total_units else None
        )

        total_rooms = self._clamp(int(round(median_rooms)) if median_rooms is not None else 6, 4, 11)
        bedrooms = self._clamp(max(total_rooms // 2, 2), 2, 6)

        # OverallQual: calibrated with both home value AND median income tier
        value_tier = int((median_home_value or 200000) / 150000)
        income_tier = int((median_income or 60000) / 75000)
        overall_qual = self._clamp(4 + value_tier + income_tier, 4, 10)

        overall_cond = self._clamp(
            5 + (1 if (median_year_built or 1980) >= 1995 else 0),
            4,
            9,
        )

        # GarageCars: owner-occupancy rate is a strong predictor of garage availability
        if owner_rate is not None:
            if owner_rate >= 0.65:
                garage_cars = 2          # mostly owners, suburban → 2-car
            elif owner_rate >= 0.40:
                garage_cars = 1          # mixed tenure → 1-car
            else:
                garage_cars = 1          # mostly renters → shared or 1-car
        else:
            garage_cars = 2 if (median_home_value or 0) >= 250000 else 1

        # LotArea: density-adjusted (denser areas = smaller lots)
        base_lot = self._clamp(int((median_home_value or 180000) / 20), 4500, 18000)
        if persons_per_unit is not None:
            if persons_per_unit < 2.0:
                density_mult = 0.55      # urban / high-turnover tract
            elif persons_per_unit >= 3.0:
                density_mult = 1.30      # suburban / family-oriented tract
            else:
                density_mult = 1.0
            lot_area = self._clamp(int(base_lot * density_mult), 2000, 25000)
        else:
            lot_area = base_lot

        # GrLivArea: income bonus captures higher-income areas' larger floor plans
        income_sqft_bonus = int((median_income or 60000) / 1500) if median_income else 0
        gr_liv_area = self._clamp(total_rooms * 340 + income_sqft_bonus, 900, 3200)

        # HouseStyle: renter-dominated / urban tracts skew to 2-story / multi-unit forms
        house_style = "2Story" if (owner_rate is not None and owner_rate < 0.45) else "1Story"

        # Build the partial payload so PropertyType classifier can see all signals
        partial: dict = {
            "LotArea": lot_area,
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "YearBuilt": self._clamp(median_year_built or 1985, 1960, 2022),
            "YearRemodAdd": self._clamp((median_year_built or 1985) + 10, 1970, 2024),
            "GrLivArea": gr_liv_area,
            "FullBath": 2 if total_rooms >= 6 else 1,
            "HalfBath": 1 if total_rooms >= 7 else 0,
            "BedroomAbvGr": bedrooms,
            "TotRmsAbvGrd": total_rooms,
            "Fireplaces": 1 if (median_home_value or 0) >= 300000 else 0,
            "GarageCars": garage_cars,
            "GarageArea": garage_cars * 260,
            "Neighborhood": geography["name"].split(",", maxsplit=1)[0],
            "HouseStyle": house_style,
            # ── new model features surfaced from census context ──────────
            "CensusMedianValue": median_home_value,
            "MedianIncomeK": round(median_income / 1000.0, 1) if median_income else None,
            "OwnerOccupiedRate": round(owner_rate, 3) if owner_rate is not None else None,
            # NeighborhoodScore is computed at training/inference time by
            # NeighborhoodScoreService (KNN); set None so providers don't guess.
            "NeighborhoodScore": None,
            # ── enrichment context (observability, not consumed by the model) ──
            "census_median_income": median_income,
            "census_median_rent": median_rent,
            "census_owner_occupancy_rate": round(owner_rate, 3) if owner_rate is not None else None,
            "census_rent_burden_pct": rent_burden_pct,
            "census_tract_population": tract_population,
        }
        partial["PropertyType"] = classify_property_type(partial)
        return partial

    @staticmethod
    def _build_feature_provenance(
        fallback_response: ProviderResponseContract | None,
        geography: dict[str, str] | None,
        used_census: bool,
    ) -> dict[str, object]:
        providers = ["census_context"] if used_census else []
        fallback_provider_name = fallback_response.provider_name if fallback_response is not None else None
        if fallback_provider_name is not None:
            providers.append(fallback_provider_name)

        provenance: dict[str, object] = {
            "strategy": "census_context" if used_census else "fallback_only",
            "providers": providers,
            "used_census": used_census,
            "used_backfill": fallback_response is not None,
        }
        if fallback_provider_name is not None:
            provenance["backfill_provider"] = fallback_provider_name
        if geography is not None:
            provenance["census_geography"] = geography
        return provenance

    @staticmethod
    def _safe_int(value: str | None) -> int | None:
        if value is None or value.startswith("-"):
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    @staticmethod
    def _safe_float(value: str | None) -> float | None:
        if value is None or value.startswith("-"):
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(value, maximum))