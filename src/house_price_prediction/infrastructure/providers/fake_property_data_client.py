from __future__ import annotations

from datetime import UTC, datetime
import hashlib

from house_price_prediction.domain.contracts.prediction_contracts import (
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.property_type_classifier import (
    classify_property_type,
)


class FakePropertyDataClient:
    def fetch_property_features(
        self,
        normalized_address: NormalizedAddress,
    ) -> ProviderResponseContract:
        seed = normalized_address.formatted_address
        payload = {
            "LotArea": self._number(seed, "LotArea", 4500, 18000),
            "OverallQual": self._number(seed, "OverallQual", 4, 10),
            "OverallCond": self._number(seed, "OverallCond", 4, 9),
            "YearBuilt": self._number(seed, "YearBuilt", 1960, 2022),
            "YearRemodAdd": self._number(seed, "YearRemodAdd", 1970, 2024),
            "GrLivArea": self._number(seed, "GrLivArea", 900, 3200),
            "FullBath": self._number(seed, "FullBath", 1, 3),
            "HalfBath": self._number(seed, "HalfBath", 0, 2),
            "BedroomAbvGr": self._number(seed, "BedroomAbvGr", 2, 6),
            "TotRmsAbvGrd": self._number(seed, "TotRmsAbvGrd", 4, 11),
            "Fireplaces": self._number(seed, "Fireplaces", 0, 2),
            "GarageCars": self._number(seed, "GarageCars", 0, 3),
            "GarageArea": self._number(seed, "GarageArea", 0, 900),
            "Neighborhood": self._choice(
                seed,
                "Neighborhood",
                ["CollgCr", "NAmes", "OldTown", "Edwards", "Somerst"],
            ),
            "HouseStyle": self._choice(seed, "HouseStyle", ["1Story", "2Story", "SLvl"]),
            # ── new model features ──────────────────────────────────
            "CensusMedianValue": self._number(seed, "CensusMedianValue", 80_000, 800_000),
            "MedianIncomeK": round(
                30.0 + self._fraction(seed, "MedianIncomeK") * 70.0, 1
            ),
            "OwnerOccupiedRate": round(
                0.25 + self._fraction(seed, "OwnerOccupiedRate") * 0.70, 3
            ),
            # NeighborhoodScore is KNN-computed at pipeline time; fake emits None
            "NeighborhoodScore": None,
            "feature_source": "fake",
            "feature_provenance": {
                "strategy": "deterministic_fake",
                "providers": ["fake_property_data"],
                "seed": normalized_address.formatted_address,
            },
        }
        payload["PropertyType"] = classify_property_type(payload)
        return ProviderResponseContract(
            provider_name="fake_property_data",
            status="success",
            payload=payload,
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _number(seed: str, name: str, minimum: int, maximum: int) -> int:
        digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
        scale = int(digest[:8], 16) / 0xFFFFFFFF
        return minimum + int(scale * (maximum - minimum))

    @staticmethod
    def _choice(seed: str, name: str, values: list[str]) -> str:
        digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(values)
        return values[index]

    @staticmethod
    def _fraction(seed: str, name: str) -> float:
        """Return a deterministic float in [0.0, 1.0) derived from seed + name."""
        digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF