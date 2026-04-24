"""
ingest_csv_training_data.py

Maps raw CSV datasets into the canonical 20-feature JSONL format used by
scripts/train.py and scripts/build_training_pipeline.py.

Sources
-------
king-county   data/raw/Housing.csv   (21,613 rows, King County WA)
              Has: real sale prices, lat/lon, grade, condition, sqft, bedrooms,
                   bathrooms, floors, year built.
              Estimated: GarageCars, GarageArea, Fireplaces, TotRmsAbvGrd
              Absent:    CensusMedianValue, MedianIncomeK, OwnerOccupiedRate
                         (the live API Census provider fills these at inference)

ames          data/raw/housing.csv   (500 rows, Ames Iowa)
              Already in the 20-feature schema format, no mapping needed.

NeighborhoodScore
-----------------
Fitted exclusively on King County (lat, lon, actual sale price).
Using actual transaction prices as the KNN signal is non-circular:
these are real market observations, not model predictions.
The fitted NeighborhoodScoreService is saved to --model-dir for use at
inference time.  Ames rows receive None (no lat/lon in that CSV).

Generalization beyond King County
----------------------------------
The model is price-calibrated on King County data.  It generalizes to other
US addresses because:
  - Structural features (squft → GrLivArea, grade → OverallQual, bedrooms,
    year built) are universal predictors whose price relationships generalize.
  - NeighborhoodScore degrades gracefully for faraway addresses: the Gaussian
    decay produces near-zero weights at distance >> decay_km, so the score
    approaches 50 (neutral).  The live Census features (CensusMedianValue,
    MedianIncomeK, OwnerOccupiedRate) then carry the market-calibration signal.
  - PropertyType is rule-based and works for any address.
To improve national price coverage: add more regional CSVs (Zillow Research,
ATTOM, HUD) with lat/lon + sale price and re-run this script.

King County column mapping
--------------------------
KC column        Schema column     Notes
sqft_lot         LotArea
grade (1-13)     OverallQual       linear remap to 1-10
condition (1-5)  OverallCond       linear remap to 1-9
yr_built         YearBuilt
yr_renovated     YearRemodAdd      0 → yr_built (never renovated)
sqft_living      GrLivArea
bathrooms        FullBath + HalfBath  decimal split
bedrooms         BedroomAbvGr
floors           HouseStyle        1.0→1Story, 1.5→SLvl, ≥2→2Story
zipcode          Neighborhood
grade            GarageCars        estimated: <5→0, 5-7→1, 8-10→2, 11+→3
grade            Fireplaces        estimated: <9→0, 9-10→1, 11+→2
bedrooms/sqft    TotRmsAbvGrd      estimated: bedrooms+2+round(sqft_above/350)
lat/long         NeighborhoodScore KNN LOO on actual prices
—                CensusMedianValue None — live Census API provides at inference
—                MedianIncomeK     None
—                OwnerOccupiedRate None
—                PropertyType      rule-based classifier

Outputs  (--output-dir, default data/processed/)
  csv_training_data.jsonl   Canonicalized rows for scripts/train.py
  csv_ingest_report.json    Row counts, quality flags, scorer diagnostics

Scorer output  (--model-dir, default models/)
  neighborhood_scorer.joblib   Fitted KNN scorer for inference time

Usage
-----
  python scripts/ingest_csv_training_data.py
  python scripts/ingest_csv_training_data.py --source king-county
  python scripts/ingest_csv_training_data.py --source ames
  python scripts/ingest_csv_training_data.py --source both \\
      --kc-csv data/raw/Housing.csv --ames-csv data/raw/housing.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# src/ package imports
from house_price_prediction.application.services.neighborhood_score_service import (
    NeighborhoodScoreService,
)
from house_price_prediction.infrastructure.providers.property_type_classifier import (
    classify_property_type,
)

# ---------------------------------------------------------------------------
# King County column mapping helpers
# ---------------------------------------------------------------------------

def _kc_grade_to_overall_qual(grade: int) -> int:
    """Map KC building grade 1-13 → OverallQual 1-10 (linear remap)."""
    return max(1, min(10, round(grade * 10 / 13)))


def _kc_condition_to_overall_cond(condition: int) -> int:
    """Map KC condition 1-5 → OverallCond 1-9 (linear remap: 1,3,5,7,9)."""
    return 1 + (int(condition) - 1) * 2


def _kc_bathrooms_split(bathrooms: float) -> tuple[int, int]:
    """Split KC decimal bathrooms → (FullBath, HalfBath).
    e.g. 2.5 → (2, 1),  2.0 → (2, 0),  1.75 → (1, 1)
    """
    full = int(bathrooms)
    half = 1 if (bathrooms - full) >= 0.25 else 0
    return full, half


def _kc_floors_to_housestyle(floors: float) -> str:
    if floors <= 1.0:
        return "1Story"
    if floors <= 1.5:
        return "SLvl"
    return "2Story"


def _kc_estimate_total_rooms(bedrooms: int, sqft_above: int) -> int:
    """Estimate TotRmsAbvGrd from bedrooms + above-grade sqft.
    Rule of thumb: ~350 sqft per room above grade.
    """
    estimated = int(bedrooms) + 2 + round(int(sqft_above) / 350)
    return max(4, min(14, estimated))


def _kc_estimate_garage_cars(grade: int) -> int:
    if grade < 5:
        return 0
    if grade <= 7:
        return 1
    if grade <= 10:
        return 2
    return 3


def _kc_estimate_fireplaces(grade: int) -> int:
    if grade < 9:
        return 0
    if grade < 11:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Per-row mapping
# ---------------------------------------------------------------------------

def _map_kc_row(row: pd.Series, min_price: float, max_bedrooms: int) -> dict[str, Any] | None:
    """Map one King County CSV row to the training schema dict.
    Returns None to drop the row (outlier / bad data).
    Internal keys _lat and _lon are used for KNN scoring and stripped before output.
    """
    try:
        price = float(row["price"])
        bedrooms = int(row["bedrooms"])
        sqft_living = int(row["sqft_living"])
        sqft_lot = int(row["sqft_lot"])
        sqft_above = int(row.get("sqft_above", sqft_living))
        grade = int(row["grade"])
        condition = int(row["condition"])
        bathrooms = float(row["bathrooms"])
        floors = float(row["floors"])
        yr_built = int(row["yr_built"])
        yr_renovated = int(row.get("yr_renovated", 0))
        zipcode = str(int(row["zipcode"]))
        lat = float(row["lat"])
        lon = float(row["long"])
    except (KeyError, TypeError, ValueError):
        return None

    # Outlier guards: prices below threshold, unrealistic bedroom counts, tiny homes
    if price < min_price:
        return None
    if bedrooms > max_bedrooms or bedrooms < 1:
        return None
    if sqft_living < 200:
        return None

    full_bath, half_bath = _kc_bathrooms_split(bathrooms)
    overall_qual = _kc_grade_to_overall_qual(grade)
    overall_cond = _kc_condition_to_overall_cond(condition)
    house_style = _kc_floors_to_housestyle(floors)
    tot_rms = _kc_estimate_total_rooms(bedrooms, sqft_above)
    garage_cars = _kc_estimate_garage_cars(grade)
    garage_area = garage_cars * 240
    fireplaces = _kc_estimate_fireplaces(grade)
    year_remod = yr_renovated if yr_renovated > 0 else yr_built

    # PropertyType classifier uses the already-mapped structural features.
    # CensusMedianValue / OwnerOccupiedRate are not in the KC CSV so we pass
    # None — the classifier degrades gracefully (defaults: owner_rate=0.7).
    property_type = classify_property_type({
        "OverallQual": overall_qual,
        "GrLivArea": sqft_living,
        "TotRmsAbvGrd": tot_rms,
        "BedroomAbvGr": bedrooms,
        "HouseStyle": house_style,
        "CensusMedianValue": None,
        "OwnerOccupiedRate": None,
    })

    return {
        "LotArea": sqft_lot,
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": yr_built,
        "YearRemodAdd": year_remod,
        "GrLivArea": sqft_living,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BedroomAbvGr": bedrooms,
        "TotRmsAbvGrd": tot_rms,
        "Fireplaces": fireplaces,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "PropertyType": property_type,
        "HouseStyle": house_style,
        "NeighborhoodScore": None,   # filled in after KNN fitting
        "CensusMedianValue": None,   # absent from KC CSV; Census API at inference
        "MedianIncomeK": None,       # absent from KC CSV
        "OwnerOccupiedRate": None,   # absent from KC CSV
        "Neighborhood": zipcode,
        "SalePrice": price,
        "_lat": lat,
        "_lon": lon,
        "_source": "king-county",
    }


def _map_ames_row(row: pd.Series, min_price: float) -> dict[str, Any] | None:
    """Map one Ames CSV row (already in schema format) to the training dict."""
    try:
        price = float(row["SalePrice"])
        if price < min_price:
            return None

        property_type = classify_property_type({
            "OverallQual": row["OverallQual"],
            "GrLivArea": row["GrLivArea"],
            "TotRmsAbvGrd": row["TotRmsAbvGrd"],
            "BedroomAbvGr": row["BedroomAbvGr"],
            "HouseStyle": str(row["HouseStyle"]),
            "CensusMedianValue": None,
            "OwnerOccupiedRate": None,
        })

        return {
            "LotArea": int(row["LotArea"]),
            "OverallQual": int(row["OverallQual"]),
            "OverallCond": int(row["OverallCond"]),
            "YearBuilt": int(row["YearBuilt"]),
            "YearRemodAdd": int(row["YearRemodAdd"]),
            "GrLivArea": int(row["GrLivArea"]),
            "FullBath": int(row["FullBath"]),
            "HalfBath": int(row["HalfBath"]),
            "BedroomAbvGr": int(row["BedroomAbvGr"]),
            "TotRmsAbvGrd": int(row["TotRmsAbvGrd"]),
            "Fireplaces": int(row["Fireplaces"]),
            "GarageCars": int(row["GarageCars"]),
            "GarageArea": int(row["GarageArea"]),
            "PropertyType": property_type,
            "HouseStyle": str(row["HouseStyle"]),
            "NeighborhoodScore": None,       # no lat/lon in Ames CSV
            "CensusMedianValue": None,
            "MedianIncomeK": None,
            "OwnerOccupiedRate": None,
            "Neighborhood": str(row["Neighborhood"]),
            "SalePrice": price,
            "_lat": None,
            "_lon": None,
            "_source": "ames",
        }
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# NeighborhoodScore: KNN LOO on KC actual prices
# ---------------------------------------------------------------------------

def _fit_and_assign_neighborhood_scores(
    rows: list[dict[str, Any]],
    k: int,
    decay_km: float,
    scorer_path: Path,
) -> NeighborhoodScoreService | None:
    """Fit scorer on KC rows (actual prices as signal) and assign LOO scores.

    Using actual sale prices as the KNN signal is non-circular because they
    are real market transactions, not predictions from the current model.
    The LOO method prevents a row from influencing its own score (no leakage).
    """
    # Only rows from King County have lat/lon
    valid_indices = [
        i for i, r in enumerate(rows)
        if (
            r.get("_lat") is not None
            and r.get("_lon") is not None
            and r.get("SalePrice") is not None
            and np.isfinite(r["_lat"])
            and np.isfinite(r["_lon"])
            and r["SalePrice"] > 0
        )
    ]

    if not valid_indices:
        print("  [scorer] No geocoded rows — NeighborhoodScore left as None for all rows.")
        return None

    lats = [rows[i]["_lat"] for i in valid_indices]
    lons = [rows[i]["_lon"] for i in valid_indices]
    prices = [rows[i]["SalePrice"] for i in valid_indices]

    print(f"  [scorer] Fitting KNN scorer on {len(lats):,} King County rows ...")
    svc = NeighborhoodScoreService(k=k, decay_km=decay_km)
    svc.fit(lats, lons, prices)

    print("  [scorer] Computing leave-one-out scores (no label leakage) ...")
    loo_scores = svc.score_loo_batch()

    # loo_scores has same length as valid_indices because fit() received the
    # same rows in the same order (all are finite, no internal drops)
    for local_i, global_i in enumerate(valid_indices):
        rows[global_i]["NeighborhoodScore"] = round(loo_scores[local_i], 2)

    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    svc.save(scorer_path)
    print(f"  [scorer] Saved to {scorer_path}")
    return svc


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_king_county(path: Path, min_price: float, max_bedrooms: int) -> list[dict[str, Any]]:
    print(f"  [kc] Reading {path} ...")
    df = pd.read_csv(path)
    print(f"  [kc] {len(df):,} raw rows")

    rows: list[dict[str, Any]] = []
    dropped = 0
    for _, row in df.iterrows():
        mapped = _map_kc_row(row, min_price=min_price, max_bedrooms=max_bedrooms)
        if mapped is None:
            dropped += 1
        else:
            rows.append(mapped)

    print(f"  [kc] {len(rows):,} rows mapped  ({dropped} dropped as outliers)")
    return rows


def _load_ames(path: Path, min_price: float) -> list[dict[str, Any]]:
    print(f"  [ames] Reading {path} ...")
    df = pd.read_csv(path)
    print(f"  [ames] {len(df):,} raw rows")

    rows: list[dict[str, Any]] = []
    dropped = 0
    for _, row in df.iterrows():
        mapped = _map_ames_row(row, min_price=min_price)
        if mapped is None:
            dropped += 1
        else:
            rows.append(mapped)

    print(f"  [ames] {len(rows):,} rows mapped  ({dropped} dropped)")
    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_INTERNAL_KEYS = {"_lat", "_lon", "_source"}


def _strip_internal(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in _INTERNAL_KEYS}


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_strip_internal(row)) + "\n")
    print(f"  Wrote {len(rows):,} rows → {path}")


def _write_report(
    report: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Wrote report  → {path}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(rows: list[dict[str, Any]], scorer: NeighborhoodScoreService | None) -> None:
    kc_rows = [r for r in rows if r.get("_source") == "king-county"]
    ames_rows = [r for r in rows if r.get("_source") == "ames"]
    scored_rows = [r for r in rows if r.get("NeighborhoodScore") is not None]

    prices = [r["SalePrice"] for r in rows]
    prop_types = {}
    for r in rows:
        pt = r.get("PropertyType", "unknown")
        prop_types[pt] = prop_types.get(pt, 0) + 1

    print()
    print("=" * 58)
    print("  INGEST SUMMARY")
    print("=" * 58)
    print(f"  Total rows          : {len(rows):,}")
    print(f"  King County rows    : {len(kc_rows):,}")
    print(f"  Ames rows           : {len(ames_rows):,}")
    print(f"  Rows with NeighScore: {len(scored_rows):,}")
    if prices:
        print(f"  Price range         : ${min(prices):,.0f} – ${max(prices):,.0f}")
        print(f"  Median price        : ${float(np.median(prices)):,.0f}")
    print(f"  PropertyType dist   : {prop_types}")
    if scorer:
        diag = scorer.diagnostics()
        print(f"  Scorer ref points   : {diag.get('reference_point_count', 0):,}")
        print(f"  Scorer k / decay    : {diag.get('k')} / {diag.get('decay_km')} km")
    print("  Census features     : None (CensusMedianValue, MedianIncomeK,")
    print("                        OwnerOccupiedRate — filled by live API")
    print("=" * 58)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map raw CSV datasets to the canonical 20-feature training JSONL format.",
    )
    parser.add_argument(
        "--source",
        choices=["king-county", "ames", "both"],
        default="both",
        help="Which CSV source(s) to ingest (default: both).",
    )
    parser.add_argument(
        "--kc-csv",
        default="data/raw/Housing.csv",
        help="Path to King County dataset (default: data/raw/Housing.csv).",
    )
    parser.add_argument(
        "--ames-csv",
        default="data/raw/housing.csv",
        help="Path to Ames Iowa dataset (default: data/raw/housing.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for JSONL and report (default: data/processed).",
    )
    parser.add_argument(
        "--output-file",
        default="csv_training_data.jsonl",
        help="Output JSONL filename (default: csv_training_data.jsonl).",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory for neighborhood_scorer.joblib (default: models).",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=10,
        help="KNN neighbour count for NeighborhoodScore (default: 10).",
    )
    parser.add_argument(
        "--knn-decay-km",
        type=float,
        default=8.0,
        help="Gaussian decay radius in km (default: 8.0).",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=50_000.0,
        help="Drop rows with SalePrice below this threshold (default: 50000).",
    )
    parser.add_argument(
        "--max-bedrooms",
        type=int,
        default=10,
        help="Drop KC rows with more bedrooms than this (default: 10).",
    )
    parser.add_argument(
        "--min-output-rows",
        type=int,
        default=100,
        help="Exit with an error if fewer rows are produced (default: 100).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_path = output_dir / args.output_file
    report_path = output_dir / "csv_ingest_report.json"
    scorer_path = model_dir / "neighborhood_scorer.joblib"

    print("=" * 58)
    print("  CSV TRAINING DATA INGESTION")
    print(f"  source   : {args.source}")
    print(f"  output   : {output_path}")
    print(f"  scorer   : {scorer_path}")
    print("=" * 58)

    all_rows: list[dict[str, Any]] = []

    if args.source in ("king-county", "both"):
        kc_path = Path(args.kc_csv)
        if not kc_path.exists():
            print(f"ERROR: King County CSV not found at {kc_path}", file=sys.stderr)
            sys.exit(1)
        all_rows.extend(
            _load_king_county(kc_path, min_price=args.min_price, max_bedrooms=args.max_bedrooms)
        )

    if args.source in ("ames", "both"):
        ames_path = Path(args.ames_csv)
        if not ames_path.exists():
            print(f"ERROR: Ames CSV not found at {ames_path}", file=sys.stderr)
            sys.exit(1)
        all_rows.extend(_load_ames(ames_path, min_price=args.min_price))

    if len(all_rows) < args.min_output_rows:
        print(
            f"ERROR: Only {len(all_rows)} rows produced; "
            f"minimum required is {args.min_output_rows}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fit scorer and assign NeighborhoodScores to all rows with lat/lon
    scorer = _fit_and_assign_neighborhood_scores(
        all_rows,
        k=args.knn_k,
        decay_km=args.knn_decay_km,
        scorer_path=scorer_path,
    )

    _print_summary(all_rows, scorer)

    # Write training JSONL (internal _lat/_lon/_source stripped)
    _write_jsonl(all_rows, output_path)

    # Write report
    kc_count = sum(1 for r in all_rows if r.get("_source") == "king-county")
    ames_count = sum(1 for r in all_rows if r.get("_source") == "ames")
    scored_count = sum(1 for r in all_rows if r.get("NeighborhoodScore") is not None)
    prices = [r["SalePrice"] for r in all_rows]
    prop_types = {}
    for r in all_rows:
        pt = r.get("PropertyType", "unknown")
        prop_types[pt] = prop_types.get(pt, 0) + 1

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": args.source,
        "output": str(output_path),
        "total_rows": len(all_rows),
        "king_county_rows": kc_count,
        "ames_rows": ames_count,
        "rows_with_neighborhood_score": scored_count,
        "price_min": float(min(prices)) if prices else None,
        "price_max": float(max(prices)) if prices else None,
        "price_median": float(np.median(prices)) if prices else None,
        "property_type_distribution": prop_types,
        "knn_k": args.knn_k,
        "knn_decay_km": args.knn_decay_km,
        "scorer_path": str(scorer_path),
        "scorer_diagnostics": scorer.diagnostics() if scorer else None,
        "census_features_present": False,
        "census_note": (
            "CensusMedianValue / MedianIncomeK / OwnerOccupiedRate are absent "
            "from both CSVs and left as None.  The live Census API fills them "
            "at inference time for any US address."
        ),
        "kc_column_mapping": {
            "grade→OverallQual": "linear remap 1-13 → 1-10",
            "condition→OverallCond": "linear remap 1-5 → 1-9",
            "bathrooms→FullBath+HalfBath": "decimal split at 0.25",
            "floors→HouseStyle": "1→1Story, 1.5→SLvl, ≥2→2Story",
            "GarageCars": "estimated from grade",
            "GarageArea": "GarageCars × 240 sqft",
            "Fireplaces": "estimated from grade",
            "TotRmsAbvGrd": "estimated from bedrooms + sqft_above / 350",
            "Neighborhood": "zipcode string",
        },
    }
    _write_report(report, report_path)

    print(f"\nDone. {len(all_rows):,} rows ready for training.")
    print(f"  Train:  RAW_DATA_PATH={output_path} python scripts/train.py --min-rows=100")


if __name__ == "__main__":
    main()
