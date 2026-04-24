"""
NeighborhoodScoreService

Computes a location-aware neighborhood price tier score (0–100) for any
(lat, lon) point by finding its K nearest geocoded reference properties and
returning a distance-decay-weighted average of their census median home values.

Why this matters
----------------
The old ``Neighborhood`` feature was a city-name string that created huge
OOV (out-of-vocabulary) problems when new cities appeared in live traffic.
The ``NeighborhoodScore`` is a continuous numeric signal rooted in real
census data (ACS B25077 tract median home value), giving the model a spatial
economic context that generalises to any address in the US.

Signal properties
-----------------
- **Non-circular**: derived from Census ACS data, NOT from the model's own
  predicted_price.  The KNN reference values come from external public data.
- **Stable**: Census ACS data changes at most once per year; score drift is
  gradual and detectable via gap analysis.
- **Inference-compatible**: the scorer can be persisted (joblib) alongside
  the model artifact and loaded at API startup for real-time scoring.

Scoring formula
---------------
For query point q:
  distances_km  = haversine(q, reference_points[k_nearest])
  weights       = exp(-0.5 * (distances_km / decay_km)^2)  # Gaussian decay
  raw_score     = weighted_average(reference_values, weights)
  score_0_100   = clamp((raw_score - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * 100, 0, 100)

Usage
-----
Training pipeline (build once per run):
  svc = NeighborhoodScoreService(k=10, decay_km=8.0)
  svc.fit(lats, lons, census_median_values)
  scores = [svc.score(lat, lon) for lat, lon in zip(train_lats, train_lons)]
  svc.save(Path("models/neighborhood_scorer.joblib"))

Inference (load once at startup):
  svc = NeighborhoodScoreService.load(Path("models/neighborhood_scorer.joblib"))
  score = svc.score(property_lat, property_lon)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# National range used for 0-100 normalisation.
# US census tract median home values run roughly $50k … $2M.
# We clip at 1.2M to keep the score meaningful in HCOL markets.
_NORMALISE_MIN = 50_000.0
_NORMALISE_MAX = 1_200_000.0

# Earth radius in km (WGS-84 mean)
_EARTH_RADIUS_KM = 6371.0

# Fallback score when there is no reference data or no coordinates
SCORE_FALLBACK = 50.0


def _haversine_km(
    lat1_rad: float,
    lon1_rad: float,
    lat2_rad: np.ndarray,
    lon2_rad: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine distance in km between one query and N references."""
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class NeighborhoodScoreService:
    """
    KNN-based neighbourhood price tier scorer.

    Parameters
    ----------
    k : int
        Number of nearest reference points to use per query (default 10).
    decay_km : float
        Gaussian decay distance in km.  At this distance the weight is ~60 %
        of the weight at zero distance.  Default 8 km gives a suburban-scale
        influence radius.
    """

    def __init__(self, k: int = 10, decay_km: float = 8.0) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        if decay_km <= 0:
            raise ValueError("decay_km must be > 0")
        self.k = k
        self.decay_km = decay_km
        self._lats_rad: np.ndarray | None = None
        self._lons_rad: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit / build reference index
    # ------------------------------------------------------------------

    def fit(
        self,
        lats: list[float] | np.ndarray,
        lons: list[float] | np.ndarray,
        census_median_values: list[float] | np.ndarray,
    ) -> "NeighborhoodScoreService":
        """
        Build the reference index from known geocoded properties.

        Parameters
        ----------
        lats, lons : array-like of float
            WGS-84 decimal degrees.
        census_median_values : array-like of float
            Census ACS median home value (B25077) for each point in dollars.
        """
        lats_arr = np.asarray(lats, dtype=np.float64)
        lons_arr = np.asarray(lons, dtype=np.float64)
        vals_arr = np.asarray(census_median_values, dtype=np.float64)

        if lats_arr.shape != lons_arr.shape or lats_arr.shape != vals_arr.shape:
            raise ValueError("lats, lons, and census_median_values must have the same length.")
        if lats_arr.ndim != 1:
            raise ValueError("Input arrays must be 1-D.")

        # Drop rows with NaN coordinates or values
        valid_mask = (
            np.isfinite(lats_arr)
            & np.isfinite(lons_arr)
            & np.isfinite(vals_arr)
            & (vals_arr > 0)
        )
        if valid_mask.sum() == 0:
            logger.warning("NeighborhoodScoreService.fit: no valid reference points.")
            return self

        self._lats_rad = np.radians(lats_arr[valid_mask])
        self._lons_rad = np.radians(lons_arr[valid_mask])
        self._values = vals_arr[valid_mask]
        self._fitted = True
        logger.info(
            "NeighborhoodScoreService fitted with %d reference points "
            "(k=%d, decay_km=%.1f).",
            int(valid_mask.sum()),
            self.k,
            self.decay_km,
        )
        return self

    # ------------------------------------------------------------------
    # Score a single location
    # ------------------------------------------------------------------

    def score(self, lat: float | None, lon: float | None) -> float:
        """
        Return the neighbourhood price tier score in [0, 100].

        Returns ``SCORE_FALLBACK`` when:
        - The scorer has not been fitted.
        - lat/lon are not available.
        - Fewer than 1 valid reference point exists.
        """
        if not self._fitted or self._lats_rad is None:
            return SCORE_FALLBACK
        if lat is None or lon is None or not (np.isfinite(lat) and np.isfinite(lon)):
            return SCORE_FALLBACK

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        distances_km = _haversine_km(lat_rad, lon_rad, self._lats_rad, self._lons_rad)

        # Take the k nearest (or all available points if fewer than k)
        k_eff = min(self.k, len(distances_km))
        nn_idx = np.argpartition(distances_km, kth=min(k_eff - 1, len(distances_km) - 1))[:k_eff]
        nn_distances = distances_km[nn_idx]
        nn_values = self._values[nn_idx]

        # Gaussian decay weighting
        weights = np.exp(-0.5 * (nn_distances / self.decay_km) ** 2)
        total_weight = weights.sum()
        if total_weight == 0:
            raw_score = float(np.mean(nn_values))
        else:
            raw_score = float(np.dot(weights, nn_values) / total_weight)

        # Normalise to 0-100
        score = (raw_score - _NORMALISE_MIN) / (_NORMALISE_MAX - _NORMALISE_MIN) * 100.0
        return float(np.clip(score, 0.0, 100.0))

    def score_batch(
        self,
        lats: list[float | None],
        lons: list[float | None],
    ) -> list[float]:
        """Score a batch of locations efficiently."""
        return [self.score(lat, lon) for lat, lon in zip(lats, lons)]

    # ------------------------------------------------------------------
    # Leave-one-out scoring (for training data — avoids label leakage)
    # ------------------------------------------------------------------

    def score_loo(self, idx: int) -> float:
        """
        Return the LOO score for training index ``idx``.

        Temporarily excludes the point at ``idx`` from the reference set so
        that the training pipeline can compute an unbiased neighbourhood score
        for each training row without leaking its own value.
        """
        if not self._fitted or self._lats_rad is None or self._values is None:
            return SCORE_FALLBACK
        if idx < 0 or idx >= len(self._lats_rad):
            return SCORE_FALLBACK

        lat_rad = self._lats_rad[idx]
        lon_rad = self._lons_rad[idx]

        # Build temporary mask excluding idx
        mask = np.ones(len(self._lats_rad), dtype=bool)
        mask[idx] = False
        if mask.sum() == 0:
            return SCORE_FALLBACK

        distances_km = _haversine_km(lat_rad, lon_rad, self._lats_rad[mask], self._lons_rad[mask])
        k_eff = min(self.k, len(distances_km))
        nn_idx = np.argpartition(distances_km, kth=min(k_eff - 1, len(distances_km) - 1))[:k_eff]
        nn_distances = distances_km[nn_idx]
        nn_values = self._values[mask][nn_idx]

        weights = np.exp(-0.5 * (nn_distances / self.decay_km) ** 2)
        total_weight = weights.sum()
        if total_weight == 0:
            raw_score = float(np.mean(nn_values))
        else:
            raw_score = float(np.dot(weights, nn_values) / total_weight)

        score = (raw_score - _NORMALISE_MIN) / (_NORMALISE_MAX - _NORMALISE_MIN) * 100.0
        return float(np.clip(score, 0.0, 100.0))

    def score_loo_batch(self) -> list[float]:
        """
        Return LOO scores for every point in the reference set.

        This is the correct method to use when adding ``NeighborhoodScore``
        to training data so that no point influences its own score.
        """
        if not self._fitted or self._lats_rad is None:
            return []
        return [self.score_loo(i) for i in range(len(self._lats_rad))]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        if not self._fitted or self._values is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "reference_point_count": int(len(self._values)),
            "k": self.k,
            "decay_km": self.decay_km,
            "value_stats": {
                "min": float(self._values.min()),
                "mean": float(self._values.mean()),
                "median": float(np.median(self._values)),
                "max": float(self._values.max()),
            },
            "normalise_range": [_NORMALISE_MIN, _NORMALISE_MAX],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist the scorer to a joblib file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "k": self.k,
            "decay_km": self.decay_km,
            "lats_rad": self._lats_rad,
            "lons_rad": self._lons_rad,
            "values": self._values,
            "fitted": self._fitted,
        }
        joblib.dump(state, path)
        logger.info("NeighborhoodScoreService saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "NeighborhoodScoreService":
        """Load a previously persisted scorer."""
        if not path.exists():
            raise FileNotFoundError(f"Neighbourhood scorer not found at {path}.")
        state = joblib.load(path)
        svc = cls(k=state["k"], decay_km=state["decay_km"])
        svc._lats_rad = state["lats_rad"]
        svc._lons_rad = state["lons_rad"]
        svc._values = state["values"]
        svc._fitted = state["fitted"]
        return svc

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_training_jsonl(
        cls,
        path: Path,
        lat_col: str = "lat",
        lon_col: str = "lon",
        value_col: str = "CensusMedianValue",
        k: int = 10,
        decay_km: float = 8.0,
    ) -> "NeighborhoodScoreService":
        """
        Build scorer from a JSONL file that has lat/lon and census value columns.

        The file is expected to have been produced by the training pipeline with
        the ``normalized_address`` lat/lon columns injected alongside features.
        """
        import pandas as pd

        if not path.exists():
            raise FileNotFoundError(f"Training JSONL not found at {path}.")
        df = pd.read_json(path, lines=True)
        for col in (lat_col, lon_col, value_col):
            if col not in df.columns:
                logger.warning(
                    "Column '%s' not found in %s — cannot fit NeighborhoodScoreService.",
                    col,
                    path,
                )
                return cls(k=k, decay_km=decay_km)

        valid = df[[lat_col, lon_col, value_col]].dropna()
        svc = cls(k=k, decay_km=decay_km)
        svc.fit(
            lats=valid[lat_col].tolist(),
            lons=valid[lon_col].tolist(),
            census_median_values=valid[value_col].tolist(),
        )
        return svc

    @classmethod
    def from_candidates(
        cls,
        candidates: list[dict[str, Any]],
        k: int = 10,
        decay_km: float = 8.0,
    ) -> "NeighborhoodScoreService":
        """
        Build scorer from the raw API candidate list emitted by
        ``/v1/meta/live-feature-candidates``.

        Each candidate is expected to have:
          ``normalized_address.latitude``, ``normalized_address.longitude``
          ``features.CensusMedianValue``
        """
        lats: list[float] = []
        lons: list[float] = []
        values: list[float] = []

        for item in candidates:
            addr = item.get("normalized_address") or {}
            feats = item.get("features") or {}
            lat = addr.get("latitude")
            lon = addr.get("longitude")
            val = feats.get("CensusMedianValue")

            # Accept heuristic fallback if census value unavailable
            if val is None:
                overall_qual = feats.get("OverallQual", 5)
                try:
                    val = float(overall_qual) * 40_000.0
                except (TypeError, ValueError):
                    val = 200_000.0

            try:
                lats.append(float(lat))
                lons.append(float(lon))
                values.append(float(val))
            except (TypeError, ValueError):
                continue

        svc = cls(k=k, decay_km=decay_km)
        if lats:
            svc.fit(lats=lats, lons=lons, census_median_values=values)
        return svc
