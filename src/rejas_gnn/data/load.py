"""Loading and validation of the raw street-segment table (one row per OSM edge)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from rejas_gnn.config import DataConfig

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "nodo_origen", "nodo_destino", "lat_origen", "lon_origen", "lat_destino", "lon_destino",
    "tipo_via_osm", "estado_cierre", "distancia_m",
)
VALID_STATES: frozenset[str] = frozenset({"abierta", "cierre", "condominio", "informal", "medio"})
LABEL_LEAK_COLUMNS: frozenset[str] = frozenset(
    {"estado_cierre", "estado_cierre_codigo", "es_cierre", "es_abierta", "es_condominio", "y"}
)


def load_segments(cfg: DataConfig) -> pd.DataFrame:
    """Read the raw edge table, validate its schema and derive geometry and labels.

    Adds columns ``arista_id``, ``x_m``, ``y_m`` (UTM 19S midpoints), ``y`` (binary label) and
    ``is_condominio``. Duplicate ``(nodo_origen, nodo_destino)`` pairs are kept (as in the
    original study) but logged.

    Raises
    ------
    FileNotFoundError, ValueError
        On missing file, missing columns, unknown closure states or NaN coordinates.
    """
    path = Path(cfg.raw_edges_csv)
    if not path.exists():
        raise FileNotFoundError(f"Raw edge file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    df["estado_cierre"] = df["estado_cierre"].astype(str).str.strip().str.lower()
    unknown = set(df["estado_cierre"]) - VALID_STATES
    if unknown:
        raise ValueError(f"Unknown estado_cierre values: {sorted(unknown)}")
    coords = ["lat_origen", "lon_origen", "lat_destino", "lon_destino"]
    if df[coords].isna().any().any():
        raise ValueError("NaN coordinates found in raw edges")

    n_dup = int(df.duplicated(["nodo_origen", "nodo_destino"]).sum())
    if n_dup:
        logger.warning("%d duplicated (origen, destino) pairs kept as separate segments", n_dup)

    if cfg.drop_condominio:
        before = len(df)
        df = df[df["estado_cierre"] != "condominio"].reset_index(drop=True)
        logger.info("Dropped %d condominio segments", before - len(df))

    df["arista_id"] = (
        df["nodo_origen"].astype(str) + "_" + df["nodo_destino"].astype(str) + "_0"
    )
    lat_mid = (df["lat_origen"] + df["lat_destino"]) / 2.0
    lon_mid = (df["lon_origen"] + df["lon_destino"]) / 2.0
    tr = Transformer.from_crs(cfg.crs_geographic, cfg.crs_metric, always_xy=True)
    x, y = tr.transform(lon_mid.to_numpy(), lat_mid.to_numpy())
    df["x_m"] = np.asarray(x, dtype=np.float64)
    df["y_m"] = np.asarray(y, dtype=np.float64)
    df["y"] = df["estado_cierre"].isin(cfg.positive_states).astype(np.int64)
    df["is_condominio"] = (df["estado_cierre"] == "condominio").astype(np.int64)

    assert df["y"].isin((0, 1)).all()
    assert np.isfinite(df[["x_m", "y_m"]].to_numpy()).all(), "non-finite projected coordinates"
    logger.info(
        "Loaded %d segments | positives=%d (%.2f%%) | condominio=%d",
        len(df), int(df["y"].sum()), 100 * df["y"].mean(), int(df["is_condominio"].sum()),
    )
    return df


def attach_covariates(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Left-join external segment-level covariates (census 2024, MBHT) by ``arista_id``.

    Only columns starting with ``cfg.covariate_prefixes`` are taken; label-derived columns
    are refused defensively. Missing file -> returns ``df`` unchanged (logged).
    """
    if cfg.covariates_csv is None or not Path(cfg.covariates_csv).exists():
        logger.warning("Covariate file not found; continuing with network/geometry features only")
        return df
    cov = pd.read_csv(cfg.covariates_csv, low_memory=False)
    if "arista_id" not in cov.columns:
        raise ValueError("Covariate file must contain 'arista_id'")
    keep = [c for c in cov.columns if c.startswith(cfg.covariate_prefixes)]
    leaked = LABEL_LEAK_COLUMNS.intersection(keep)
    assert not leaked, f"Label columns leaked into covariates: {sorted(leaked)}"
    cov = cov[["arista_id", *keep]].drop_duplicates("arista_id")
    out = df.merge(cov, on="arista_id", how="left", validate="m:1")
    assert len(out) == len(df), "merge changed row count"
    logger.info("Attached %d covariate columns from %s", len(keep), Path(cfg.covariates_csv).name)
    return out
