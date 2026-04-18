from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# Canonical aliases help keep training data stable while provider payloads evolve.
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
    "Neighborhood": ("neighborhood_name",),
    "HouseStyle": ("house_style",),
}

NUMERIC_FEATURES: frozenset[str] = frozenset(
    {
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
    }
)


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


def _canonicalize_feature_map(
    source_features: dict[str, Any], expected_features: list[str]
) -> tuple[dict[str, Any], set[str]]:
    canonical: dict[str, Any] = {}
    consumed_keys: set[str] = set()

    for feature_name in expected_features:
        alias_candidates = (feature_name, *FEATURE_ALIASES.get(feature_name, ()))
        value = _first_present_value(source_features, alias_candidates)
        canonical[feature_name] = _coerce_feature_value(feature_name, value)
        for alias in alias_candidates:
            if alias in source_features:
                consumed_keys.add(alias)

    unknown_keys = {key for key in source_features if key not in consumed_keys}
    return canonical, unknown_keys


def _fetch_capabilities(base_url: str) -> dict:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/v1/meta/capabilities", timeout=30)
    except requests.RequestException as exc:
        raise SystemExit(
            f"Could not reach API capabilities endpoint at {base_url.rstrip('/')}/v1/meta/capabilities: {exc}"
        ) from exc
    response.raise_for_status()
    return response.json()


def _fetch_candidate_page(
    base_url: str,
    limit: int,
    offset: int,
    min_completeness_score: float,
    include_reused: bool,
) -> dict:
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/v1/meta/live-feature-candidates",
            params={
                "limit": limit,
                "offset": offset,
                "min_completeness_score": min_completeness_score,
                "include_reused": str(include_reused).lower(),
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise SystemExit(
            "Could not fetch live feature candidates from API: "
            f"{base_url.rstrip('/')}/v1/meta/live-feature-candidates ({exc})"
        ) from exc
    response.raise_for_status()
    return response.json()


def _build_training_frame(
    candidates: list[dict],
    expected_features: list[str],
    target_column: str = "SalePrice",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict] = []
    unknown_feature_keys: set[str] = set()
    invalid_target_rows = 0
    for item in candidates:
        feature_map = item.get("features", {})
        canonical_features, unknown_keys = _canonicalize_feature_map(feature_map, expected_features)
        unknown_feature_keys.update(unknown_keys)

        target_value = item.get("predicted_price")
        if target_value is None:
            invalid_target_rows += 1
            continue
        try:
            canonical_target = float(target_value)
        except (TypeError, ValueError):
            invalid_target_rows += 1
            continue

        row = dict(canonical_features)
        row[target_column] = canonical_target
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        diagnostics = {
            "input_candidate_count": len(candidates),
            "rows_with_invalid_target": invalid_target_rows,
            "rows_dropped_missing_required": 0,
            "unknown_source_feature_keys": sorted(unknown_feature_keys),
        }
        return frame, diagnostics

    # Drop rows that are missing any required model feature value.
    before_drop_count = len(frame)
    frame = frame.dropna(subset=expected_features)
    dropped_missing_required = before_drop_count - len(frame)
    frame[target_column] = frame[target_column].astype(float)

    diagnostics = {
        "input_candidate_count": len(candidates),
        "rows_with_invalid_target": invalid_target_rows,
        "rows_dropped_missing_required": dropped_missing_required,
        "unknown_source_feature_keys": sorted(unknown_feature_keys),
    }
    return frame, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap training data from live API feature candidates (no CSV source).",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="data/processed/live_feature_store.jsonl")
    parser.add_argument("--min-completeness-score", type=float, default=0.9)
    parser.add_argument("--include-reused", action="store_true")
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument(
        "--min-output-rows",
        type=int,
        default=1,
        help="Fail when extracted valid rows are below this threshold.",
    )
    parser.add_argument("--target-column", default="SalePrice")
    parser.add_argument(
        "--snapshot-dir",
        default="",
        help="Optional directory for timestamped JSONL and metadata snapshots.",
    )
    parser.add_argument(
        "--snapshot-prefix",
        default="live_feature_store",
        help="Filename prefix used for snapshot files.",
    )
    parser.add_argument(
        "--metadata-output",
        default="",
        help="Optional path to write bootstrap provenance metadata JSON.",
    )
    args = parser.parse_args()

    capabilities = _fetch_capabilities(args.base_url)
    expected_features: list[str] = capabilities.get("model_expected_features", [])
    if not expected_features:
        raise SystemExit("API did not return model_expected_features; cannot build feature store.")

    all_items: list[dict] = []
    offset = 0
    while len(all_items) < args.max_rows:
        page = _fetch_candidate_page(
            base_url=args.base_url,
            limit=min(args.page_size, args.max_rows - len(all_items)),
            offset=offset,
            min_completeness_score=args.min_completeness_score,
            include_reused=args.include_reused,
        )
        items = page.get("items", [])
        if not items:
            break
        all_items.extend(items)
        offset += len(items)

    dataset, diagnostics = _build_training_frame(
        candidates=all_items,
        expected_features=expected_features,
        target_column=args.target_column,
    )
    if dataset.empty:
        raise SystemExit(
            "No valid live feature rows found. "
            "Run more live predictions or lower --min-completeness-score."
        )
    if len(dataset) < args.min_output_rows:
        raise SystemExit(
            "Extracted live rows below minimum threshold: "
            f"required {args.min_output_rows}, found {len(dataset)}. "
            "Run more live predictions, lower --min-completeness-score, or reduce --min-output-rows."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(output_path, orient="records", lines=True)

    metadata_output_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else output_path.with_suffix(output_path.suffix + ".meta.json")
    )
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url.rstrip("/"),
        "output": str(output_path),
        "row_count": int(len(dataset)),
        "column_count": int(len(dataset.columns)),
        "expected_feature_count": int(len(expected_features)),
        "target_column": args.target_column,
        "min_completeness_score": float(args.min_completeness_score),
        "include_reused": bool(args.include_reused),
        "page_size": int(args.page_size),
        "max_rows": int(args.max_rows),
        "min_output_rows": int(args.min_output_rows),
        "input_candidate_count": int(diagnostics["input_candidate_count"]),
        "rows_with_invalid_target": int(diagnostics["rows_with_invalid_target"]),
        "rows_dropped_missing_required": int(diagnostics["rows_dropped_missing_required"]),
        "unknown_source_feature_keys": diagnostics["unknown_source_feature_keys"],
        "capabilities_contract_version": capabilities.get("contract_version"),
        "model_name": capabilities.get("model_name"),
        "model_version": capabilities.get("model_version"),
        "feature_policy_name": capabilities.get("feature_policy_name"),
        "feature_policy_version": capabilities.get("feature_policy_version"),
        "expected_features": expected_features,
    }

    snapshot_data_path: Path | None = None
    snapshot_metadata_path: Path | None = None
    if args.snapshot_dir:
        snapshot_dir = Path(args.snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        snapshot_base = f"{args.snapshot_prefix}_{snapshot_stamp}"
        snapshot_data_path = snapshot_dir / f"{snapshot_base}.jsonl"
        snapshot_metadata_path = snapshot_dir / f"{snapshot_base}.meta.json"
        dataset.to_json(snapshot_data_path, orient="records", lines=True)
        metadata["snapshot_output"] = str(snapshot_data_path)

    metadata_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if snapshot_metadata_path is not None:
        snapshot_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Live feature store written to: {output_path}")
    print(f"Metadata written to: {metadata_output_path}")
    if snapshot_data_path is not None and snapshot_metadata_path is not None:
        print(f"Snapshot data written to: {snapshot_data_path}")
        print(f"Snapshot metadata written to: {snapshot_metadata_path}")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Features: {len(expected_features)}")


if __name__ == "__main__":
    main()