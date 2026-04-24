"""
seed_live_predictions.py

Sends a geographically diverse set of US addresses to the running API to
seed the live feature store with real geocoding and census neighborhood data.

This script is an API integration test AND a live data seeder.  Each
successful request stores a feature snapshot in the database, which is then
available via /v1/meta/live-feature-candidates for bootstrap_training_data.py.

What each request captures
--------------------------
  - Real lat/lon from Nominatim geocoding
  - Real Census ACS neighborhood context (CensusMedianValue, MedianIncomeK,
    OwnerOccupiedRate) from the Census API for that tract
  - Heuristic structural estimates (GrLivArea, OverallQual, etc.)
    These are address-hash-derived approximations — NOT real property data.

Important
---------
Structural features (bedrooms, sqft, year built) from this seeder come from
the HeuristicPropertyDataClient, NOT from a real property database.  For real
structural features use ingest_csv_training_data.py (King County / Ames CSV).

The seeder's value is in building up geographic diversity in the live feature
store — real lat/lon + real census context — so the NeighborhoodScoreService
and census signals work correctly for non-KC US addresses.

Rate limiting
-------------
Nominatim requires at least 1 request per second.  The default --delay is
1.5 s to stay within policy.  Increase it if you see 429 errors.

Pre-requisites
--------------
The API must be running with real providers:
  make run-api-live
Or equivalently:
  APP_ENV=test ENABLE_MOCK_PREDICTOR=false \\
      GEOCODING_PROVIDER=free-fallback \\
      PROPERTY_DATA_PROVIDER=free-fallback \\
      DATABASE_URL=sqlite:///data/processed/real_validation.db \\
      uvicorn house_price_prediction.api.main:app --port 8000

Usage
-----
  python scripts/seed_live_predictions.py
  python scripts/seed_live_predictions.py --base-url http://127.0.0.1:8000
  python scripts/seed_live_predictions.py --delay 2.0 --output data/processed/seed_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Address list — geographically diverse US public/municipal addresses.
# Using public buildings (city halls, libraries) ensures reliable geocoding
# without using private residences.  Coverage spans 8 geographic regions.
# ---------------------------------------------------------------------------

_SEED_ADDRESSES: list[dict[str, str]] = [
    # Pacific Northwest — closest to King County training data; best KNN scores
    {"address_line_1": "600 4th Ave",            "city": "Seattle",       "state": "WA", "postal_code": "98104"},
    {"address_line_1": "450 110th Ave NE",        "city": "Bellevue",      "state": "WA", "postal_code": "98004"},
    {"address_line_1": "123 5th Ave",             "city": "Kirkland",      "state": "WA", "postal_code": "98033"},
    {"address_line_1": "3002 Colby Ave",          "city": "Everett",       "state": "WA", "postal_code": "98201"},
    {"address_line_1": "1221 SW 4th Ave",         "city": "Portland",      "state": "OR", "postal_code": "97204"},
    {"address_line_1": "1221 SW 2nd Ave",         "city": "Portland",      "state": "OR", "postal_code": "97204"},

    # West Coast
    {"address_line_1": "1 Dr Carlton B Goodlett Pl", "city": "San Francisco", "state": "CA", "postal_code": "94102"},
    {"address_line_1": "200 N Spring St",         "city": "Los Angeles",   "state": "CA", "postal_code": "90012"},
    {"address_line_1": "1200 3rd Ave",            "city": "San Diego",     "state": "CA", "postal_code": "92101"},
    {"address_line_1": "915 I St",                "city": "Sacramento",    "state": "CA", "postal_code": "95814"},

    # Mountain West
    {"address_line_1": "1437 Bannock St",         "city": "Denver",        "state": "CO", "postal_code": "80202"},
    {"address_line_1": "451 S State St",          "city": "Salt Lake City","state": "UT", "postal_code": "84111"},
    {"address_line_1": "200 W Washington St",     "city": "Phoenix",       "state": "AZ", "postal_code": "85003"},
    {"address_line_1": "1 E 1st St",              "city": "Reno",          "state": "NV", "postal_code": "89505"},

    # Sun Belt — Texas
    {"address_line_1": "301 W 2nd St",            "city": "Austin",        "state": "TX", "postal_code": "78701"},
    {"address_line_1": "901 Bagby St",            "city": "Houston",       "state": "TX", "postal_code": "77002"},
    {"address_line_1": "1500 Marilla St",         "city": "Dallas",        "state": "TX", "postal_code": "75201"},
    {"address_line_1": "203 S St Marys St",       "city": "San Antonio",   "state": "TX", "postal_code": "78205"},

    # Sun Belt — Florida
    {"address_line_1": "3500 Pan American Dr",    "city": "Miami",         "state": "FL", "postal_code": "33133"},
    {"address_line_1": "400 S Orange Ave",        "city": "Orlando",       "state": "FL", "postal_code": "32801"},
    {"address_line_1": "1 City Hall Plaza",       "city": "Tampa",         "state": "FL", "postal_code": "33602"},
    {"address_line_1": "City Hall",               "city": "Jacksonville",  "state": "FL", "postal_code": "32202"},

    # Southeast
    {"address_line_1": "55 Trinity Ave SW",       "city": "Atlanta",       "state": "GA", "postal_code": "30303"},
    {"address_line_1": "1 Public Square",         "city": "Nashville",     "state": "TN", "postal_code": "37201"},
    {"address_line_1": "600 E 4th St",            "city": "Charlotte",     "state": "NC", "postal_code": "28202"},
    {"address_line_1": "222 W Hargett St",        "city": "Raleigh",       "state": "NC", "postal_code": "27601"},
    {"address_line_1": "701 E Broad St",          "city": "Richmond",      "state": "VA", "postal_code": "23219"},

    # Midwest
    {"address_line_1": "121 N LaSalle St",        "city": "Chicago",       "state": "IL", "postal_code": "60602"},
    {"address_line_1": "90 W Broad St",           "city": "Columbus",      "state": "OH", "postal_code": "43215"},
    {"address_line_1": "350 S 5th St",            "city": "Minneapolis",   "state": "MN", "postal_code": "55415"},
    {"address_line_1": "2 Woodward Ave",          "city": "Detroit",       "state": "MI", "postal_code": "48226"},
    {"address_line_1": "414 E 12th St",           "city": "Kansas City",   "state": "MO", "postal_code": "64106"},
    {"address_line_1": "200 W Washington St",     "city": "Indianapolis",  "state": "IN", "postal_code": "46204"},
    {"address_line_1": "751 N 9th St",            "city": "Milwaukee",     "state": "WI", "postal_code": "53233"},
    {"address_line_1": "1 City Hall Plaza",       "city": "St. Louis",     "state": "MO", "postal_code": "63103"},

    # Northeast
    {"address_line_1": "1 City Hall Sq",          "city": "Boston",        "state": "MA", "postal_code": "02201"},
    {"address_line_1": "1 Penn Square",           "city": "Philadelphia",  "state": "PA", "postal_code": "19107"},
    {"address_line_1": "1350 Pennsylvania Ave NW","city": "Washington",    "state": "DC", "postal_code": "20004"},
    {"address_line_1": "260 Broadway",            "city": "New York",      "state": "NY", "postal_code": "10007"},
    {"address_line_1": "100 N Holliday St",       "city": "Baltimore",     "state": "MD", "postal_code": "21202"},
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post_prediction(
    base_url: str,
    address: dict[str, str],
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/v1/predictions"
    payload = {**address, "country": "US"}
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        return resp.status_code, resp.json()
    except requests.Timeout:
        return 0, {"error": "timeout"}
    except requests.RequestException as exc:
        return 0, {"error": str(exc)}
    except ValueError:
        return resp.status_code, {"error": "invalid JSON response"}


def _check_api_health(base_url: str, timeout: float) -> bool:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the live feature store with geographically diverse US addresses.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds between requests (Nominatim rate limit is 1/s; default: 1.5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request HTTP timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write per-address results as JSONL.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error instead of continuing.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  LIVE PREDICTION SEEDER")
    print(f"  API     : {args.base_url}")
    print(f"  Delay   : {args.delay}s between requests (Nominatim policy)")
    print(f"  Note    : structural features come from HeuristicPropertyDataClient")
    print(f"            — census context (lat/lon, neighborhood signals) is real")
    print("=" * 60)

    # Health check
    if not _check_api_health(args.base_url, timeout=10.0):
        print()
        print("ERROR: API does not appear to be running.", file=sys.stderr)
        print("       Start it with:  make run-api-live", file=sys.stderr)
        sys.exit(1)
    print("  API health: OK\n")

    results: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0

    for i, address in enumerate(_SEED_ADDRESSES, start=1):
        label = f"{address['address_line_1']}, {address['city']}, {address['state']}"
        print(f"[{i:2}/{len(_SEED_ADDRESSES)}] {label}")

        status_code, body = _post_prediction(args.base_url, address, timeout=args.timeout)

        predicted_price = body.get("predicted_price")
        feature_source = (
            body.get("feature_snapshot", {}).get("features", {}).get("feature_source")
            or body.get("provider_summary", {}).get("feature_source")
            or "—"
        )
        completeness = body.get("feature_snapshot", {}).get("completeness_score")

        if status_code in (200, 201) and predicted_price is not None:
            succeeded += 1
            flag = "✓"
            price_str = f"${predicted_price:,.0f}" if isinstance(predicted_price, (int, float)) else str(predicted_price)
            score_str = f"  completeness={completeness:.2f}" if isinstance(completeness, float) else ""
            print(f"         {flag} {price_str}  source={feature_source}{score_str}")
        else:
            failed += 1
            flag = "✗"
            err = body.get("detail") or body.get("error") or f"HTTP {status_code}"
            print(f"         {flag} FAILED: {err}")
            if args.fail_fast:
                print("Stopping (--fail-fast set).", file=sys.stderr)
                sys.exit(1)

        results.append({
            "address": address,
            "status_code": status_code,
            "predicted_price": predicted_price,
            "feature_source": feature_source,
            "completeness_score": completeness,
            "ok": status_code in (200, 201) and predicted_price is not None,
        })

        # Respect Nominatim rate limit between requests (skip after last)
        if i < len(_SEED_ADDRESSES):
            time.sleep(args.delay)

    # Summary
    print()
    print("=" * 60)
    print(f"  Sent {len(_SEED_ADDRESSES)} requests")
    print(f"  Succeeded : {succeeded}")
    print(f"  Failed    : {failed}")
    if succeeded > 0:
        print()
        print("  Next step: run bootstrap-data to extract features from DB,")
        print("  then train-from-csv for a complete training set:")
        print("    make bootstrap-data")
        print("    make train-from-csv")
    print("=" * 60)

    # Optional JSONL output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps({
                    **r,
                    "seeded_at": datetime.now(UTC).isoformat(),
                }) + "\n")
        print(f"\n  Results written to {out_path}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
