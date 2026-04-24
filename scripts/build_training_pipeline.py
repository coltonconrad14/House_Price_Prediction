"""
build_training_pipeline.py

Full live-source training data pipeline for house price prediction.

Stages
------
1. extract    Pull raw live feature candidates from the API and snapshot them.
2. enrich     Derive PropertyType from existing features; compute NeighborhoodScore
              via KNN on (lat, lon) using census median values as the scoring signal.
3. gap        Analyse completeness / distribution drift vs a reference CSV.
4. assemble   Canonicalize features, resolve labels, drop invalid rows.
5. split      Produce deterministic train / val / test splits.
6. report     Write a rich JSON report + persist the NeighborhoodScoreService.

Outputs written to --output-dir (default: data/processed/training_pipeline/)
  raw_candidates.jsonl              Raw API response items (one per line)
  training_ready.jsonl              Canonicalized features + label (full assembled set)
  splits/train.jsonl
  splits/val.jsonl
  splits/test.jsonl
  neighborhood_scorer.joblib        Persisted KNN scorer for use at inference time
  pipeline_report.json              All diagnostics, gap verdicts, readiness verdict

Label strategies (--label-source)
  predicted   Use predicted_price from live API traffic.
              CIRCULAR WARNING: the label is the current model's output.
              The pipeline flags label collapse when variance is near-zero.
              Use this only to audit feature gaps, NOT to improve the model.

Neighbourhood scoring
  NeighborhoodScore is computed by the NeighborhoodScoreService using a
  leave-one-out KNN over the assembled training set.  The scorer is saved to
  disk alongside the training data so it can be loaded at inference time.
  Signal: CensusMedianValue (ACS B25077) - non-circular, external data source.

Usage
-----
  python scripts/build_training_pipeline.py \\
      --base-url http://127.0.0.1:8000 \\
      --reference-csv data/raw/Housing.csv \\
      --output-dir data/processed/training_pipeline
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from house_price_prediction.application.services.neighborhood_score_service import (
    NeighborhoodScoreService,
)
from house_price_prediction.feature_schema import DEFAULT_PREDICTION_FEATURES
from house_price_prediction.infrastructure.providers.property_type_classifier import (
    classify_property_type,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COLUMN = "SalePrice"

NUMERIC_FEATURES: frozenset[str] = frozenset(
    {
        "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
        "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
        "Fireplaces", "GarageCars", "GarageArea",
        "NeighborhoodScore", "CensusMedianValue", "MedianIncomeK", "OwnerOccupiedRate",
    }
)

FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "LotArea": ("lot_area", "lotarea"),
    "OverallQual": ("overall_quality", "overallqual"),
    "OverallCond": ("overall_condition", "overallcond"),
    "YearBuilt": ("year_built", "yearbuilt"),
    "YearRemodAdd": ("year_remod_add", "yearremodadd", "year_remodeled"),
    "GrLivArea": ("gr_liv_area", "grlivarea", "living_area"),
    "FullBath": ("full_bath", "fullbath"),
    "HalfBath": ("half_bath", "halfbath"),
    "BedroomAbvGr": ("bedroom_abv_gr", "bedrooms", "bedroom_count"),
    "TotRmsAbvGrd": ("tot_rms_abv_grd", "total_rooms", "rooms"),
    "Fireplaces": ("fire_places", "fireplace_count"),
    "GarageCars": ("garage_cars", "garagecars"),
    "GarageArea": ("garage_area", "garagearea"),
    "Neighborhood": ("neighborhood_name", "census_tract_name"),
    "HouseStyle": ("house_style",),
    "PropertyType": ("property_type",),
    "NeighborhoodScore": ("neighborhood_score", "neighbourhood_score"),
    "CensusMedianValue": ("census_median_value", "census_median_home_value", "median_home_value"),
    "MedianIncomeK": ("median_income_k", "census_median_income_k"),
    "OwnerOccupiedRate": ("owner_occupied_rate", "census_owner_occupancy_rate", "owner_rate"),
}

_HIGH_NULL_RATE = 0.50
_MEDIUM_NULL_RATE = 0.20
_LABEL_COLLAPSE_CV = 0.01


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureGap:
    feature: str
    severity: str
    kind: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 1: Extract
# ---------------------------------------------------------------------------

def _fetch_capabilities(base_url: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/meta/capabilities"
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        raise SystemExit(f"Cannot reach capabilities endpoint at {url}: {exc}") from exc
    resp.raise_for_status()
    return resp.json()


def _fetch_candidate_page(
    base_url: str, limit: int, offset: int,
    min_completeness_score: float, include_reused: bool,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/meta/live-feature-candidates"
    try:
        resp = requests.get(
            url,
            params={
                "limit": limit, "offset": offset,
                "min_completeness_score": min_completeness_score,
                "include_reused": str(include_reused).lower(),
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise SystemExit(f"Cannot fetch live feature candidates from {url}: {exc}") from exc
    resp.raise_for_status()
    return resp.json()


def stage_extract(
    base_url: str, min_completeness_score: float, include_reused: bool,
    page_size: int, max_rows: int,
) -> list[dict[str, Any]]:
    print("[1/6] EXTRACT  pulling live feature candidates from API ...")
    all_items: list[dict[str, Any]] = []
    offset = 0
    while len(all_items) < max_rows:
        batch_limit = min(page_size, max_rows - len(all_items))
        page = _fetch_candidate_page(
            base_url=base_url, limit=batch_limit, offset=offset,
            min_completeness_score=min_completeness_score, include_reused=include_reused,
        )
        items = page.get("items", [])
        if not items:
            break
        all_items.extend(items)
        offset += len(items)
        print(f"         fetched {len(all_items)} rows ...")
    print(f"         total raw candidates: {len(all_items)}")
    return all_items


# ---------------------------------------------------------------------------
# Stage 2: Enrich - PropertyType + NeighborhoodScore
# ---------------------------------------------------------------------------

def _resolve_census_median_value(item: dict[str, Any]) -> float | None:
    feats = item.get("features") or {}
    val = feats.get("CensusMedianValue")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    # Heuristic fallback from OverallQual when census value unavailable
    oq = feats.get("OverallQual")
    if oq is not None:
        try:
            return float(oq) * 60_000.0
        except (TypeError, ValueError):
            pass
    return None


def stage_enrich(
    raw_candidates: list[dict[str, Any]],
    expected_features: list[str],
    knn_k: int,
    knn_decay_km: float,
) -> tuple[list[dict[str, Any]], NeighborhoodScoreService, dict[str, Any]]:
    """
    Enriches each candidate with:
    - PropertyType: deterministic classification from existing feature signals
    - NeighborhoodScore: leave-one-out KNN score using CensusMedianValue as the
      spatial price-tier signal (non-circular, derived from census ACS data)
    """
    print("[2/6] ENRICH   computing PropertyType + NeighborhoodScore (KNN LOO) ...")

    # Build the KNN reference from all candidates with geocoordinates
    lats: list[float] = []
    lons: list[float] = []
    census_values: list[float] = []
    lat_lon_per_item: list[tuple[float | None, float | None]] = []

    for item in raw_candidates:
        addr = item.get("normalized_address") or {}
        lat = addr.get("latitude")
        lon = addr.get("longitude")
        lat_lon_per_item.append((lat, lon))
        val = _resolve_census_median_value(item)
        if lat is not None and lon is not None and val is not None:
            try:
                lats.append(float(lat))
                lons.append(float(lon))
                census_values.append(float(val))
            except (TypeError, ValueError):
                pass

    scorer = NeighborhoodScoreService(k=knn_k, decay_km=knn_decay_km)
    if lats:
        scorer.fit(lats=lats, lons=lons, census_median_values=census_values)

    # Map (lat, lon) -> scorer reference index for LOO scoring
    ref_lat_lon_to_idx: dict[tuple[float, float], int] = {}
    ref_idx = 0
    for item in raw_candidates:
        addr = item.get("normalized_address") or {}
        lat = addr.get("latitude")
        lon = addr.get("longitude")
        val = _resolve_census_median_value(item)
        if lat is not None and lon is not None and val is not None:
            try:
                key = (float(lat), float(lon))
                if key not in ref_lat_lon_to_idx:
                    ref_lat_lon_to_idx[key] = ref_idx
                ref_idx += 1
            except (TypeError, ValueError):
                pass

    enriched: list[dict[str, Any]] = []
    property_type_counts: dict[str, int] = {}
    loo_scores_computed = 0
    fallback_scores = 0

    for item, (lat, lon) in zip(raw_candidates, lat_lon_per_item):
        enriched_item = dict(item)
        feats = dict(item.get("features") or {})

        # PropertyType
        prop_type = feats.get("PropertyType")
        if not prop_type:
            prop_type = classify_property_type(feats)
        property_type_counts[prop_type] = property_type_counts.get(prop_type, 0) + 1
        feats["PropertyType"] = prop_type

        # NeighborhoodScore
        ns = feats.get("NeighborhoodScore")
        if ns is None or not isinstance(ns, (int, float)):
            if lat is not None and lon is not None:
                try:
                    key = (float(lat), float(lon))
                    scorer_idx = ref_lat_lon_to_idx.get(key)
                    if scorer_idx is not None and scorer._fitted:
                        ns = scorer.score_loo(scorer_idx)
                        loo_scores_computed += 1
                    else:
                        ns = scorer.score(lat, lon)
                        fallback_scores += 1
                except (TypeError, ValueError):
                    ns = scorer.score(lat, lon) if lat is not None else None
                    fallback_scores += 1
            else:
                ns = None
                fallback_scores += 1
        feats["NeighborhoodScore"] = ns

        # Surface CensusMedianValue into features if not already there
        if feats.get("CensusMedianValue") is None:
            v = _resolve_census_median_value(item)
            if v is not None:
                feats["CensusMedianValue"] = v

        enriched_item["features"] = feats
        enriched.append(enriched_item)

    diagnostics = {
        "loo_scores_computed": loo_scores_computed,
        "fallback_scores": fallback_scores,
        "property_type_distribution": property_type_counts,
        "scorer_diagnostics": scorer.diagnostics(),
    }
    print(
        f"         PropertyType distribution: {property_type_counts}\n"
        f"         NeighborhoodScore: {loo_scores_computed} LOO, "
        f"{fallback_scores} fallback"
    )
    return enriched, scorer, diagnostics


# ---------------------------------------------------------------------------
# Helpers: feature canonicalization
# ---------------------------------------------------------------------------

def _first_present_value(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return None


def _coerce_feature_value(feature_name: str, value: Any) -> Any:
    if value is None:
        return None
    if feature_name in NUMERIC_FEATURES:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    normalized = str(value).strip()
    return normalized or None


def _canonicalize_row(
    source_features: dict[str, Any],
    expected_features: list[str],
) -> tuple[dict[str, Any], set[str]]:
    canonical: dict[str, Any] = {}
    consumed: set[str] = set()
    for feat in expected_features:
        candidates = (feat, *FEATURE_ALIASES.get(feat, ()))
        value = _first_present_value(source_features, candidates)
        canonical[feat] = _coerce_feature_value(feat, value)
        for alias in candidates:
            if alias in source_features:
                consumed.add(alias)
    unknown = {k for k in source_features if k not in consumed}
    return canonical, unknown


# ---------------------------------------------------------------------------
# Stage 3: Gap analysis
# ---------------------------------------------------------------------------

def _numeric_stats(series: pd.Series) -> dict[str, Any]:
    described = series.dropna().describe()
    return {
        "count": int(described.get("count", 0)),
        "mean": float(described.get("mean", float("nan"))),
        "std": float(described.get("std", float("nan"))),
        "min": float(described.get("min", float("nan"))),
        "p25": float(described.get("25%", float("nan"))),
        "p50": float(described.get("50%", float("nan"))),
        "p75": float(described.get("75%", float("nan"))),
        "max": float(described.get("max", float("nan"))),
    }


def _categorical_stats(series: pd.Series) -> dict[str, Any]:
    counts = series.dropna().value_counts()
    return {"unique_count": int(counts.shape[0]), "top_values": counts.head(10).to_dict()}


def stage_gap_analysis(
    live_df: pd.DataFrame,
    expected_features: list[str],
    reference_df: pd.DataFrame | None,
) -> tuple[list[FeatureGap], dict[str, float], dict[str, dict[str, Any]]]:
    print("[3/6] GAP      analysing feature completeness and distribution ...")
    gaps: list[FeatureGap] = []
    null_rates: dict[str, float] = {}
    feature_stats: dict[str, dict[str, Any]] = {}

    for feat in expected_features:
        if feat not in live_df.columns:
            null_rates[feat] = 1.0
            gaps.append(FeatureGap(
                feature=feat, severity="HIGH", kind="null_rate",
                message=f"Feature '{feat}' is entirely absent from live data.",
                detail={"null_rate": 1.0},
            ))
            continue

        col = live_df[feat]
        null_rate = float(col.isna().mean())
        null_rates[feat] = null_rate

        if null_rate >= _HIGH_NULL_RATE:
            sev: str | None = "HIGH"
        elif null_rate >= _MEDIUM_NULL_RATE:
            sev = "MEDIUM"
        elif null_rate > 0:
            sev = "LOW"
        else:
            sev = None

        if sev:
            gaps.append(FeatureGap(
                feature=feat, severity=sev, kind="null_rate",
                message=f"Feature '{feat}' has {null_rate:.1%} null rate in live data.",
                detail={"null_rate": null_rate},
            ))

        if feat in NUMERIC_FEATURES:
            live_stats = _numeric_stats(col)
            feature_stats[feat] = {"type": "numeric", "live": live_stats}
        else:
            live_stats = _categorical_stats(col)
            feature_stats[feat] = {"type": "categorical", "live": live_stats}

        if reference_df is not None and feat in reference_df.columns:
            ref_col = reference_df[feat]
            if feat in NUMERIC_FEATURES:
                ref_stats = _numeric_stats(ref_col)
                feature_stats[feat]["reference"] = ref_stats
                ref_std = ref_stats["std"]
                if ref_std and ref_std > 0 and not np.isnan(live_stats["mean"]):
                    drift = abs(live_stats["mean"] - ref_stats["mean"]) / ref_std
                    feature_stats[feat]["mean_drift_sigmas"] = round(float(drift), 3)
                    dsev = "HIGH" if drift > 3.0 else ("MEDIUM" if drift > 2.0 else None)
                    if dsev:
                        gaps.append(FeatureGap(
                            feature=feat, severity=dsev, kind="distribution_drift",
                            message=(
                                f"Feature '{feat}' live mean ({live_stats['mean']:.2f}) "
                                f"deviates {drift:.1f}sigma from reference "
                                f"({ref_stats['mean']:.2f})."
                            ),
                            detail={
                                "live_mean": live_stats["mean"],
                                "ref_mean": ref_stats["mean"],
                                "drift_sigmas": float(drift),
                            },
                        ))
            else:
                ref_stats = _categorical_stats(ref_col)
                feature_stats[feat]["reference"] = ref_stats
                live_vals = set(live_df[feat].dropna().unique())
                ref_vals = set(ref_col.dropna().unique())
                unseen = sorted(live_vals - ref_vals)
                missing = sorted(ref_vals - live_vals)
                feature_stats[feat]["unseen_in_reference"] = unseen
                feature_stats[feat]["missing_from_live"] = missing[:20]
                if unseen:
                    gaps.append(FeatureGap(
                        feature=feat, severity="HIGH", kind="categorical_coverage",
                        message=(
                            f"Feature '{feat}' has {len(unseen)} value(s) in live data "
                            f"not present in reference: "
                            f"{unseen[:5]}{'...' if len(unseen) > 5 else ''}."
                        ),
                        detail={
                            "unseen_in_reference": unseen,
                            "live_unique_count": len(live_vals),
                            "ref_unique_count": len(ref_vals),
                        },
                    ))
                if missing:
                    gaps.append(FeatureGap(
                        feature=feat, severity="MEDIUM", kind="categorical_coverage",
                        message=(
                            f"Feature '{feat}' is missing {len(missing)} reference value(s): "
                            f"{missing[:5]}{'...' if len(missing) > 5 else ''}."
                        ),
                        detail={
                            "missing_from_live": missing,
                            "live_unique_count": len(live_vals),
                            "ref_unique_count": len(ref_vals),
                        },
                    ))

    by_sev: dict[str, int] = {}
    for g in gaps:
        by_sev[g.severity] = by_sev.get(g.severity, 0) + 1
    print(f"         gaps found: {by_sev or 'none'}")
    return gaps, null_rates, feature_stats


# ---------------------------------------------------------------------------
# Stage 4: Assemble
# ---------------------------------------------------------------------------

def _detect_label_collapse(series: pd.Series) -> bool:
    mean = series.mean()
    if mean == 0:
        return series.std() == 0
    return float(series.std() / abs(mean)) < _LABEL_COLLAPSE_CV


def stage_assemble(
    enriched_candidates: list[dict[str, Any]],
    expected_features: list[str],
    label_source: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[FeatureGap]]:
    print("[4/6] ASSEMBLE canonicalizing features and resolving labels ...")
    rows: list[dict[str, Any]] = []
    unknown_keys: set[str] = set()
    invalid_target = 0
    extra_gaps: list[FeatureGap] = []

    for item in enriched_candidates:
        source_features = item.get("features", {})
        canonical, unknown = _canonicalize_row(source_features, expected_features)
        unknown_keys.update(unknown)

        raw_label = item.get("predicted_price") if label_source == "predicted" else None
        if raw_label is None:
            invalid_target += 1
            continue
        try:
            label_value = float(raw_label)
        except (TypeError, ValueError):
            invalid_target += 1
            continue

        row = dict(canonical)
        row[TARGET_COLUMN] = label_value
        # Pass lat/lon through so the scorer can be rebuilt from this JSONL
        addr = item.get("normalized_address") or {}
        row["lat"] = addr.get("latitude")
        row["lon"] = addr.get("longitude")
        rows.append(row)

    assembled = pd.DataFrame(rows)
    if assembled.empty:
        raise SystemExit(
            "No valid rows assembled.\n"
            "  - Ensure the API has processed real prediction requests.\n"
            "  - Lower --min-completeness-score."
        )

    before = len(assembled)
    assembled = assembled.dropna(subset=expected_features)
    dropped_missing = before - len(assembled)

    label_series = assembled[TARGET_COLUMN]
    collapse = _detect_label_collapse(label_series)
    if collapse:
        cv = (
            float(label_series.std() / abs(label_series.mean()))
            if label_series.mean() != 0 else 0.0
        )
        extra_gaps.append(FeatureGap(
            feature=TARGET_COLUMN,
            severity="CRITICAL",
            kind="label_collapse",
            message=(
                f"Label '{TARGET_COLUMN}' has near-zero variation (CV={cv:.4f}). "
                "All predicted_prices are essentially identical. "
                "Training on this data will produce a constant predictor. "
                "This is the CIRCULAR LABEL problem: --label-source predicted reuses "
                "the model's own output as the training target. "
                "You need real sale price labels to improve model quality."
            ),
            detail={
                "mean": float(label_series.mean()),
                "std": float(label_series.std()),
                "cv": cv,
                "unique_values": int(label_series.nunique()),
            },
        ))

    diagnostics = {
        "rows_assembled": len(assembled),
        "rows_dropped_invalid_target": invalid_target,
        "rows_dropped_missing_features": dropped_missing,
        "unknown_source_feature_keys": sorted(unknown_keys),
        "label_stats": _numeric_stats(label_series),
        "label_collapse_detected": collapse,
    }
    print(
        f"         assembled rows: {len(assembled)} "
        f"(dropped {dropped_missing} missing-features, {invalid_target} invalid-label)"
    )
    if collapse:
        print("         CRITICAL: label collapse detected")
    return assembled, diagnostics, extra_gaps


# ---------------------------------------------------------------------------
# Stage 5: Split
# ---------------------------------------------------------------------------

def stage_split(
    assembled: pd.DataFrame,
    train_ratio: float, val_ratio: float, random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[5/6] SPLIT    producing train / val / test splits ...")
    if round(1.0 - train_ratio - val_ratio, 6) < 0:
        raise ValueError("train_ratio + val_ratio must not exceed 1.0")
    shuffled = assembled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n = len(shuffled)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    train_df = shuffled.iloc[:n_train]
    val_df = shuffled.iloc[n_train : n_train + n_val]
    test_df = shuffled.iloc[n_train + n_val :]
    print(f"         train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Stage 6: Save + report
# ---------------------------------------------------------------------------

def _build_readiness_verdict(
    gaps: list[FeatureGap], assembled_row_count: int, min_rows: int,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    critical = [g for g in gaps if g.severity == "CRITICAL"]
    high = [g for g in gaps if g.severity == "HIGH"]
    if critical:
        verdict = "NOT_READY"
        for g in critical:
            notes.append(f"CRITICAL: {g.message}")
    elif assembled_row_count < min_rows:
        verdict = "NOT_READY"
        notes.append(
            f"Row count ({assembled_row_count}) below minimum ({min_rows}). "
            "Run more live predictions or lower --min-rows."
        )
    elif high:
        verdict = "PARTIAL"
        for g in high[:3]:
            notes.append(f"HIGH gap: {g.message}")
        if len(high) > 3:
            notes.append(f"...and {len(high) - 3} more HIGH gaps (see gaps list).")
    else:
        verdict = "READY"
        notes.append("No critical or high-severity gaps. Dataset is ready for training.")
    return verdict, notes


def stage_save_and_report(
    output_dir: Path,
    raw_candidates: list[dict[str, Any]],
    assembled: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scorer: NeighborhoodScoreService,
    gaps: list[FeatureGap],
    null_rates: dict[str, float],
    feature_stats: dict[str, dict[str, Any]],
    assemble_diagnostics: dict[str, Any],
    enrich_diagnostics: dict[str, Any],
    capabilities: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    print("[6/6] REPORT   writing outputs and pipeline report ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw_candidates.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for item in raw_candidates:
            fh.write(json.dumps(item) + "\n")

    # training_ready includes lat/lon for scorer rebuild
    training_ready_path = output_dir / "training_ready.jsonl"
    assembled.to_json(training_ready_path, orient="records", lines=True)

    # splits strip lat/lon (not model features)
    model_cols = [c for c in assembled.columns if c not in {"lat", "lon"}]
    train_df[model_cols].to_json(splits_dir / "train.jsonl", orient="records", lines=True)
    val_df[model_cols].to_json(splits_dir / "val.jsonl", orient="records", lines=True)
    test_df[model_cols].to_json(splits_dir / "test.jsonl", orient="records", lines=True)

    scorer_path = output_dir / "neighborhood_scorer.joblib"
    scorer.save(scorer_path)

    verdict, notes = _build_readiness_verdict(gaps, len(assembled), getattr(args, "min_rows", 25))

    report_dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pipeline_version": "2.0.0",
        "label_source": args.label_source,
        "circular_label_warning": (args.label_source == "predicted"),
        "label_collapse_detected": assemble_diagnostics.get("label_collapse_detected", False),
        "row_counts": {
            "extracted": len(raw_candidates),
            "assembled": len(assembled),
            "dropped_invalid_target": assemble_diagnostics.get("rows_dropped_invalid_target", 0),
            "dropped_missing_features": assemble_diagnostics.get("rows_dropped_missing_features", 0),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "enrichment": enrich_diagnostics,
        "feature_null_rates": null_rates,
        "feature_stats": feature_stats,
        "gaps": [
            {"feature": g.feature, "severity": g.severity,
             "kind": g.kind, "message": g.message, "detail": g.detail}
            for g in gaps
        ],
        "gap_summary": {
            sev: sum(1 for g in gaps if g.severity == sev)
            for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")
        },
        "readiness_verdict": verdict,
        "readiness_notes": notes,
        "unknown_source_feature_keys": assemble_diagnostics.get("unknown_source_feature_keys", []),
        "capabilities": {
            "contract_version": capabilities.get("contract_version"),
            "model_name": capabilities.get("model_name"),
            "model_version": capabilities.get("model_version"),
            "feature_policy_name": capabilities.get("feature_policy_name"),
            "model_expected_features": capabilities.get("model_expected_features", []),
        },
        "output_paths": {
            "raw_candidates": str(raw_path),
            "training_ready": str(training_ready_path),
            "train_split": str(splits_dir / "train.jsonl"),
            "val_split": str(splits_dir / "val.jsonl"),
            "test_split": str(splits_dir / "test.jsonl"),
            "neighborhood_scorer": str(scorer_path),
        },
    }

    if "label_stats" in assemble_diagnostics:
        report_dict["feature_stats"][TARGET_COLUMN] = {
            "type": "numeric",
            "live": assemble_diagnostics["label_stats"],
        }

    report_path = output_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(report_path: Path) -> None:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    rc = data["row_counts"]
    verdict = data["readiness_verdict"]
    gaps_summary = data.get("gap_summary", {})
    enrich = data.get("enrichment", {})
    pt_dist = enrich.get("property_type_distribution", {})

    print()
    print("=" * 62)
    print(" TRAINING PIPELINE SUMMARY")
    print("=" * 62)
    print(f"  Extracted rows:             {rc['extracted']}")
    print(f"  Assembled rows:             {rc['assembled']}")
    print(f"  Train / Val / Test:         {rc['train']} / {rc['val']} / {rc['test']}")
    print()
    if pt_dist:
        print("  PropertyType distribution:")
        for ptype, cnt in sorted(pt_dist.items(), key=lambda x: -x[1]):
            print(f"    {ptype:<20s} {cnt}")
    sd = enrich.get("scorer_diagnostics", {})
    if sd.get("fitted"):
        print(
            f"  NeighborhoodScore KNN:      "
            f"k={sd['k']}, decay={sd['decay_km']} km, "
            f"{sd['reference_point_count']} reference pts"
        )
    print()
    print(f"  Readiness verdict:  {verdict}")
    for note in data.get("readiness_notes", []):
        print(f"    - {note}")
    print()
    if any(gaps_summary.values()):
        print("  Gap summary:")
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            cnt = gaps_summary.get(sev, 0)
            if cnt:
                print(f"    {sev:<12s} {cnt}")
    print()
    print(f"  Full report:   {report_path}")
    scorer_path = data.get("output_paths", {}).get("neighborhood_scorer", "")
    if scorer_path:
        print(f"  KNN scorer:    {scorer_path}")
    print("=" * 62)
    if data.get("circular_label_warning"):
        print()
        print("  NOTE: label-source=predicted is CIRCULAR.")
        print("  The model trained on this data converges to its own prior.")
        print("  Real sale price labels are required to actually improve model quality.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a training-ready dataset from live API sources with "
            "PropertyType classification and KNN neighbourhood scoring."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", default="data/processed/training_pipeline")
    parser.add_argument(
        "--reference-csv", default="",
        help="Reference CSV for distribution gap analysis (not mixed into training data).",
    )
    parser.add_argument(
        "--label-source", default="predicted",
        help="'predicted' uses API predicted_price (circular warning).",
    )
    parser.add_argument("--min-completeness-score", type=float, default=0.8)
    parser.add_argument("--include-reused", action="store_true")
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-rows", type=int, default=25,
                        help="Minimum assembled rows to pass readiness check.")
    parser.add_argument("--knn-k", type=int, default=10,
                        help="K nearest neighbours for NeighborhoodScore.")
    parser.add_argument("--knn-decay-km", type=float, default=8.0,
                        help="Gaussian decay distance (km) for neighbourhood KNN.")

    args = parser.parse_args()

    if args.label_source != "predicted":
        print(
            f"ERROR: --label-source '{args.label_source}' is not yet supported.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)

    reference_df: pd.DataFrame | None = None
    if args.reference_csv:
        ref_path = Path(args.reference_csv)
        if not ref_path.exists():
            print(f"WARNING: --reference-csv path not found: {ref_path}", file=sys.stderr)
        else:
            try:
                reference_df = pd.read_csv(ref_path)
                print(f"[setup]  loaded reference CSV: {ref_path} ({len(reference_df)} rows)")
            except Exception as exc:
                print(f"WARNING: could not load reference CSV: {exc}", file=sys.stderr)

    capabilities = _fetch_capabilities(args.base_url)
    expected_features: list[str] = capabilities.get("model_expected_features", [])
    if not expected_features:
        print(
            "WARNING: API returned no model_expected_features — "
            "falling back to DEFAULT_PREDICTION_FEATURES.",
            file=sys.stderr,
        )
        expected_features = list(DEFAULT_PREDICTION_FEATURES)

    print(f"[setup]  expected features ({len(expected_features)}): {expected_features}")

    # Ensure new features are included even when the deployed model predates this pipeline
    for new_feat in ("PropertyType", "NeighborhoodScore", "CensusMedianValue",
                     "MedianIncomeK", "OwnerOccupiedRate"):
        if new_feat not in expected_features:
            expected_features.append(new_feat)
            print(f"[setup]  injected new feature into pipeline scope: {new_feat}")

    print(
        "[setup]  NOTICE: label-source=predicted is circular. "
        "Label collapse will be flagged if price variance is near-zero."
    )

    # Stage 1
    raw_candidates = stage_extract(
        base_url=args.base_url,
        min_completeness_score=args.min_completeness_score,
        include_reused=args.include_reused,
        page_size=args.page_size,
        max_rows=args.max_rows,
    )
    if not raw_candidates:
        raise SystemExit(
            "No live feature candidates returned by the API.\n"
            "  - Ensure the API is running and has processed real prediction requests.\n"
            "  - Lower --min-completeness-score."
        )

    # Stage 2
    enriched_candidates, scorer, enrich_diagnostics = stage_enrich(
        raw_candidates=raw_candidates,
        expected_features=expected_features,
        knn_k=args.knn_k,
        knn_decay_km=args.knn_decay_km,
    )

    # Build live features DataFrame for gap analysis
    live_feature_rows: list[dict[str, Any]] = []
    for item in enriched_candidates:
        canonical, _ = _canonicalize_row(item.get("features", {}), expected_features)
        live_feature_rows.append(canonical)
    live_features_df = pd.DataFrame(live_feature_rows)

    # Stage 3
    all_gaps, null_rates, feature_stats = stage_gap_analysis(
        live_df=live_features_df,
        expected_features=expected_features,
        reference_df=reference_df,
    )

    # Stage 4
    assembled, assemble_diagnostics, extra_gaps = stage_assemble(
        enriched_candidates=enriched_candidates,
        expected_features=expected_features,
        label_source=args.label_source,
    )
    all_gaps.extend(extra_gaps)

    # Stage 5
    train_df, val_df, test_df = stage_split(
        assembled=assembled,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
    )

    # Stage 6
    report_path = stage_save_and_report(
        output_dir=output_dir,
        raw_candidates=raw_candidates,
        assembled=assembled,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        scorer=scorer,
        gaps=all_gaps,
        null_rates=null_rates,
        feature_stats=feature_stats,
        assemble_diagnostics=assemble_diagnostics,
        enrich_diagnostics=enrich_diagnostics,
        capabilities=capabilities,
        args=args,
    )

    _print_summary(report_path)


if __name__ == "__main__":
    main()
