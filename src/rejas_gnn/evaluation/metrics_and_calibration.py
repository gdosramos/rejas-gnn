"""Phase-2 metrics, probability calibration and spatial export of P(closure | features).

* Discrimination: ROC-AUC, PR-AUC, F1 / precision / recall, MCC (``rejas_gnn.metrics``).
* Calibration: Brier score, Expected Calibration Error (ECE, equal-mass bins), reliability
  table, and an **isotonic calibrator fitted on validation-block predictions only**.
* Spatial: Moran's I of residuals on the graph; export of a GeoPackage/GeoJSON with the
  out-of-fold probability for 100 % of the segments.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from rejas_gnn.metrics import classification_metrics

logger = logging.getLogger(__name__)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-mass bins: Σ_b (n_b/N)·|mean(y_b) − mean(p_b)|."""
    assert y.shape == p.shape and len(y) > 0
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    ece = 0.0
    for b in bins:
        if len(b):
            ece += len(b) / len(y) * abs(y[b].mean() - p[b].mean())
    return float(ece)


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Per-bin observed frequency vs mean predicted probability (equal-mass bins)."""
    order = np.argsort(p)
    rows = []
    for i, b in enumerate(np.array_split(order, n_bins)):
        rows.append({"bin": i, "n": len(b), "p_mean": float(p[b].mean()), "y_rate": float(y[b].mean())})
    return pd.DataFrame(rows)


def fit_isotonic(y_val: np.ndarray, p_val: np.ndarray) -> IsotonicRegression:
    """Monotone calibrator fitted on validation predictions (never on test)."""
    assert len(y_val) == len(p_val) and 0 < y_val.sum() < len(y_val), "val set needs both classes"
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val, y_val)
    return iso


def full_report(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Discrimination + calibration metrics in one dictionary."""
    assert np.isfinite(p).all(), "non-finite probabilities"
    assert (p >= -1e-9).all() and (p <= 1 + 1e-9).all(), "probabilities must be in [0,1]"
    p = np.clip(p, 0.0, 1.0)
    out = classification_metrics(y, p, threshold)
    out["ece"] = expected_calibration_error(y, p)
    return out


def export_probability_map(segments: gpd.GeoDataFrame, prob: np.ndarray, out_path: Path,
                           extra: dict[str, np.ndarray] | None = None) -> Path:
    """Write a vector layer (GeoPackage or GeoJSON, by extension) with P(closure) per segment.

    The layer keeps ``arista_id``, ``estado_cierre``, ``y`` and ``p_closure``; GeoJSON is
    written in EPSG:4326 (RFC 7946), GeoPackage keeps the metric CRS.
    """
    assert len(prob) == len(segments), "one probability per segment"
    assert np.isfinite(prob).all() and (prob >= -1e-9).all() and (prob <= 1 + 1e-9).all()
    prob = np.clip(prob, 0.0, 1.0)
    cols = ["arista_id", "nodo_origen", "nodo_destino", "nombre_calle", "tipo_via_osm",
            "estado_cierre", "y", "geometry"]
    cols = [c for c in cols if c in segments.columns]
    gdf = segments[cols].copy()
    gdf["p_closure"] = prob.astype(float)
    for k, v in (extra or {}).items():
        assert len(v) == len(gdf)
        gdf[k] = v
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".geojson":
        gdf.to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")
    else:
        gdf.to_file(out_path, driver="GPKG", layer="p_closure")
    logger.info("Probability map written: %s (%d segments)", out_path, len(gdf))
    return out_path
