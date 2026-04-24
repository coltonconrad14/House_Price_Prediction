from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeaturePolicyDefinition:
    name: str
    version: str
    description: str
    emphasis_features: tuple[str, ...]
    weights: dict[str, float]


_FEATURE_POLICY_DEFINITIONS: tuple[FeaturePolicyDefinition, ...] = (
    FeaturePolicyDefinition(
        name="balanced-v1",
        version="v1",
        description="Uniform baseline weighting across all aligned model features.",
        emphasis_features=(),
        weights={},
    ),
    FeaturePolicyDefinition(
        name="quality-first-v1",
        version="v1",
        description=(
            "Emphasizes structural quality, livable area, property classification, "
            "and neighbourhood market tier for pricing."
        ),
        emphasis_features=(
            "OverallQual",
            "GrLivArea",
            "YearBuilt",
            "PropertyType",
            "NeighborhoodScore",
            "CensusMedianValue",
        ),
        weights={
            "OverallQual": 3.0,
            "GrLivArea": 2.5,
            "YearBuilt": 1.5,
            "PropertyType": 2.0,
            "NeighborhoodScore": 2.5,
            "CensusMedianValue": 2.0,
            "MedianIncomeK": 1.5,
            "OwnerOccupiedRate": 1.0,
            # legacy neighbourhood string kept for backward compat
            "Neighborhood": 1.0,
        },
    ),
    FeaturePolicyDefinition(
        name="land-first-v1",
        version="v1",
        description="Emphasize lot size, land-adjacent characteristics, and neighbourhood tier.",
        emphasis_features=(
            "LotArea",
            "Neighborhood",
            "NeighborhoodScore",
            "CensusMedianValue",
        ),
        weights={
            "LotArea": 3.0,
            "Neighborhood": 1.5,
            "NeighborhoodScore": 2.5,
            "CensusMedianValue": 2.0,
            "MedianIncomeK": 1.5,
            "OwnerOccupiedRate": 1.0,
            "PropertyType": 1.5,
        },
    ),
    FeaturePolicyDefinition(
        name="market-context-v1",
        version="v1",
        description=(
            "Prioritises real-time market context signals: neighbourhood KNN score, "
            "census economic data, and property type classification. "
            "Designed for live-data-trained models that have full census enrichment."
        ),
        emphasis_features=(
            "NeighborhoodScore",
            "CensusMedianValue",
            "MedianIncomeK",
            "PropertyType",
            "OwnerOccupiedRate",
        ),
        weights={
            "NeighborhoodScore": 4.0,
            "CensusMedianValue": 3.5,
            "MedianIncomeK": 3.0,
            "PropertyType": 3.0,
            "OwnerOccupiedRate": 2.0,
            "OverallQual": 2.0,
            "GrLivArea": 2.0,
            "LotArea": 1.5,
            "YearBuilt": 1.5,
            "Neighborhood": 1.0,
        },
    ),
)


def list_feature_policy_definitions() -> tuple[FeaturePolicyDefinition, ...]:
    return _FEATURE_POLICY_DEFINITIONS


def list_feature_policy_names() -> tuple[str, ...]:
    return tuple(policy.name for policy in _FEATURE_POLICY_DEFINITIONS)


def get_feature_policy_definition(policy_name: str) -> FeaturePolicyDefinition | None:
    normalized_name = policy_name.strip().lower()
    for policy in _FEATURE_POLICY_DEFINITIONS:
        if policy.name == normalized_name:
            return policy
    return None


def get_feature_policy_weights(policy_name: str) -> dict[str, float]:
    policy = get_feature_policy_definition(policy_name)
    if policy is None:
        logger.warning(
            "Unknown feature policy %r; falling back to 'balanced-v1'. "
            "Check FEATURE_POLICY_NAME or FEATURE_POLICY_STATE_OVERRIDES.",
            policy_name,
        )
        fallback = get_feature_policy_definition("balanced-v1")
        assert fallback is not None
        return fallback.weights
    return policy.weights
