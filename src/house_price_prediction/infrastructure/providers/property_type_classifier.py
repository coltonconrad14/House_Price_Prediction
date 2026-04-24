"""
PropertyTypeClassifier

Derives a property type label from a provider payload dict using a
deterministic rule hierarchy.  The classifier is intentionally explicit —
no ML model, no external calls — so it works at any stage of the pipeline
(provider enrichment, training data assembly, gap analysis).

Property type taxonomy
----------------------
luxury        High-quality, large or high-value properties.
single_family Standard detached / attached residential with high owner occupancy.
townhouse     Multi-level attached dwelling in mixed-tenure area.
condo         Compact unit in a predominantly renter or high-density area.
multifamily   Large unit count, lower owner-occupancy, often multiple households.

Signal hierarchy
----------------
1. Luxury gate    — quality + area + census value thresholds
2. Multifamily    — low owner-occupancy + many bedrooms/rooms (suggests multiple units)
3. Condo          — low owner-occupancy + small footprint
4. Townhouse      — 2-story + medium footprint + mixed tenure
5. Single-family  — everything else (safe default)

Inputs used (all optional; classifier degrades gracefully when absent)
----------------------------------------------------------------------
OverallQual         int 1-10
GrLivArea           int sq-ft
TotRmsAbvGrd        int
BedroomAbvGr        int
HouseStyle          str  e.g. "1Story", "2Story", "SFoyer"
CensusMedianValue   float  census tract median home value USD
OwnerOccupiedRate   float  0.0-1.0
OverallCond         int 1-10
"""
from __future__ import annotations

from typing import Any

# -----------------------------------------------------------------------
# Thresholds
# -----------------------------------------------------------------------

_LUXURY_QUAL_THRESHOLD = 9            # OverallQual >= 9 → luxury
_LUXURY_QUAL_AREA_QUAL = 8            # OverallQual >= 8 + large area → luxury
_LUXURY_QUAL_AREA_SQFT = 3_200        # GrLivArea sq-ft
_LUXURY_CENSUS_VALUE = 800_000        # census tract median → luxury market

_MULTIFAMILY_OWNER_RATE = 0.30        # very low owner-occupancy
_MULTIFAMILY_MIN_BEDROOMS = 4         # suggests multiple units at this density
_MULTIFAMILY_MIN_ROOMS = 8

_CONDO_OWNER_RATE = 0.45              # predominantly renter-occupied
_CONDO_MAX_SQFT = 1_800               # compact footprint
_CONDO_MAX_ROOMS = 6

_TOWNHOUSE_MAX_SQFT = 2_400
_TOWNHOUSE_MAX_OWNER_RATE = 0.70      # not fully owner-occupied suburb
_TOWNHOUSE_MIN_OWNER_RATE = 0.35      # some owner presence (vs rental block)
_TOWNHOUSE_STYLES = {"2story", "2.5story", "2.5fin"}


def _safe_float(value: Any | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any | None) -> str:
    return str(value).strip().lower() if value is not None else ""


def classify_property_type(payload: dict[str, Any]) -> str:
    """
    Classify a property based on the provider payload.

    Returns one of: ``"luxury"``, ``"single_family"``, ``"townhouse"``,
    ``"condo"``, ``"multifamily"``.
    """
    overall_qual = _safe_float(payload.get("OverallQual"), 5.0)
    gr_liv_area = _safe_float(payload.get("GrLivArea"), 1_500.0)
    tot_rooms = _safe_float(payload.get("TotRmsAbvGrd"), 6.0)
    bedrooms = _safe_float(payload.get("BedroomAbvGr"), 3.0)
    house_style = _safe_str(payload.get("HouseStyle") or "1story")
    census_value = _safe_float(payload.get("CensusMedianValue"), 0.0)
    owner_rate = _safe_float(payload.get("OwnerOccupiedRate"), 0.7)

    # ---------------------------------------------------------------
    # 1. Luxury gate
    # ---------------------------------------------------------------
    if overall_qual >= _LUXURY_QUAL_THRESHOLD:
        return "luxury"
    if overall_qual >= _LUXURY_QUAL_AREA_QUAL and gr_liv_area >= _LUXURY_QUAL_AREA_SQFT:
        return "luxury"
    if census_value >= _LUXURY_CENSUS_VALUE:
        return "luxury"

    # ---------------------------------------------------------------
    # 2. Multifamily
    #    Hallmarks: very low owner-occupancy AND high bedroom/room count
    #    (proxy for multiple dwelling units in a building).
    # ---------------------------------------------------------------
    if (
        owner_rate < _MULTIFAMILY_OWNER_RATE
        and bedrooms >= _MULTIFAMILY_MIN_BEDROOMS
        and tot_rooms >= _MULTIFAMILY_MIN_ROOMS
    ):
        return "multifamily"

    # ---------------------------------------------------------------
    # 3. Condo
    #    Hallmarks: low owner-occupancy AND compact footprint.
    # ---------------------------------------------------------------
    if owner_rate < _CONDO_OWNER_RATE and gr_liv_area < _CONDO_MAX_SQFT and tot_rooms <= _CONDO_MAX_ROOMS:
        return "condo"

    # ---------------------------------------------------------------
    # 4. Townhouse
    #    Hallmarks: multi-level style, medium footprint, mixed tenure.
    # ---------------------------------------------------------------
    style_clean = house_style.replace(" ", "").replace("-", "").replace(".", "")
    is_two_story = any(s in style_clean for s in _TOWNHOUSE_STYLES) or "2" in style_clean
    if (
        is_two_story
        and gr_liv_area < _TOWNHOUSE_MAX_SQFT
        and _TOWNHOUSE_MIN_OWNER_RATE <= owner_rate < _TOWNHOUSE_MAX_OWNER_RATE
    ):
        return "townhouse"

    # ---------------------------------------------------------------
    # 5. Default: single-family
    # ---------------------------------------------------------------
    return "single_family"


def property_type_from_features(features: dict[str, Any]) -> str:
    """
    Alias for ``classify_property_type`` accepting a canonicalized feature
    dict (keys match the model's expected feature names).
    """
    return classify_property_type(features)
