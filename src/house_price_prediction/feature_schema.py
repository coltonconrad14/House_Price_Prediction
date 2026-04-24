from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


DEFAULT_PREDICTION_FEATURES: tuple[str, ...] = (
    # ── structural / physical ──────────────────────────────────────────
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "GrLivArea",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageCars",
    "GarageArea",
    # ── property classification ────────────────────────────────────────
    "PropertyType",     # single_family | condo | townhouse | multifamily | luxury
    "HouseStyle",       # 1Story | 2Story | SFoyer | …
    # ── neighbourhood / market context ────────────────────────────────
    "NeighborhoodScore",     # KNN-derived price tier 0-100 (non-circular census signal)
    "CensusMedianValue",     # ACS B25077 tract median home value (USD)
    "MedianIncomeK",         # ACS B19013 tract median household income / 1000
    "OwnerOccupiedRate",     # fraction of owner-occupied units in census tract (0-1)
    "Neighborhood",          # human-readable neighbourhood / tract label (categorical)
)


def align_feature_payload(
    expected_feature_names: Iterable[str],
    source_features: Mapping[str, Any],
) -> dict[str, Any]:
    ordered_feature_names = list(expected_feature_names)
    if not ordered_feature_names:
        return dict(source_features)
    return {feature_name: source_features.get(feature_name) for feature_name in ordered_feature_names}