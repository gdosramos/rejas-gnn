"""Geometry loading, urban-graph construction and network-metric extraction (Phase 1).

Formal objects
--------------
* **Primal graph** ``G = (V, E)``: ``V`` = OSM intersections, ``E`` = street segments
  (undirected, weighted by length). Built with NetworkX for topological metrics.
* **Dual / line graph** ``L(G)``: nodes = segments of ``G``; two segments are adjacent iff they
  share an intersection. The closure label lives on segments, so the learning task is
  **node classification on L(G)** (equivalently, edge classification on ``G``).

Two feature sets are produced for the ablation protocol:
* ``"geometry"``        – Experiment A: purely topological/geometric attributes of the segment.
* ``"geometry_socio"``  – Experiment B: A + zonal socio-economic covariates (census 2024,
  MBHT 2017) attached by inverse-distance averaging of census blocks within 150 m.

Geometry source: either the tabular ``pac.csv`` (segment endpoints → straight ``LineString``)
or any GeoPandas-readable layer (GeoPackage/GeoJSON/Shapefile) with the same attribute columns
and a ``LineString`` geometry, in which case the real polyline is used for length and export.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from torch_geometric.data import Data

from rejas_gnn.config import DataConfig, GraphConfig
from rejas_gnn.data.features import (
    assemble_feature_matrix,
    build_primal_graph,
    geometry_features,
    network_features,
)
from rejas_gnn.data.graph import build_pyg_data
from rejas_gnn.data.load import attach_covariates, load_segments

logger = logging.getLogger(__name__)
SOCIO_PREFIXES: tuple[str, ...] = ("cpv24_", "uai_")


@dataclass(frozen=True)
class UrbanGraph:
    """Container returned by :func:`build_urban_graph`."""

    segments: gpd.GeoDataFrame      # one row per segment, EPSG:32719, LineString geometry
    primal: nx.Graph                # G = (intersections, segments)
    data: Data                      # PyG object on L(G) (or radius/union variant)
    feature_names: list[str]
    feature_set: str


def load_geometries(cfg: DataConfig) -> gpd.GeoDataFrame:
    """Load street segments as a validated GeoDataFrame in the metric CRS.

    Accepts the tabular CSV (endpoints only) or a vector layer. Validates: geometry type is
    ``LineString``, no empty/invalid geometries, CRS defined, required attributes present.
    """
    path = Path(cfg.raw_edges_csv)
    if path.suffix.lower() == ".csv":
        df = load_segments(cfg)
        geom = [LineString([(lo, la), (ld, lt)]) for lo, la, ld, lt in
                zip(df.lon_origen, df.lat_origen, df.lon_destino, df.lat_destino, strict=True)]
        gdf = gpd.GeoDataFrame(df, geometry=geom, crs=cfg.crs_geographic).to_crs(cfg.crs_metric)
        logger.warning("Geometry built from endpoints (straight lines); real polylines unavailable")
    else:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise ValueError(f"{path.name} has no CRS")
        tmp = gdf.drop(columns="geometry").to_csv(index=False)
        # reuse tabular validation/labelling by round-tripping the attributes
        from io import StringIO
        cfg_tab = DataConfig(**{**cfg.__dict__, "raw_edges_csv": Path("/dev/null")})
        df = pd.read_csv(StringIO(tmp))
        gdf = gpd.GeoDataFrame(df, geometry=gdf.geometry.values, crs=gdf.crs).to_crs(cfg.crs_metric)
        _label_and_project(gdf, cfg_tab)
    assert (gdf.geom_type == "LineString").all(), "all geometries must be LineString"
    assert gdf.geometry.notna().all() and (~gdf.geometry.is_empty).all(), "empty geometries"
    assert gdf.geometry.is_valid.all(), "invalid geometries"
    assert gdf.crs is not None and gdf.crs.to_epsg() == 32719, "expected EPSG:32719"
    gdf["length_geom_m"] = gdf.geometry.length
    return gdf


def _label_and_project(gdf: gpd.GeoDataFrame, cfg: DataConfig) -> None:
    """In-place: label, arista_id and midpoint coordinates for a vector layer input."""
    from rejas_gnn.data.load import VALID_STATES

    gdf["estado_cierre"] = gdf["estado_cierre"].astype(str).str.strip().str.lower()
    unknown = set(gdf["estado_cierre"]) - VALID_STATES
    if unknown:
        raise ValueError(f"Unknown estado_cierre values: {sorted(unknown)}")
    gdf["arista_id"] = gdf["nodo_origen"].astype(str) + "_" + gdf["nodo_destino"].astype(str) + "_0"
    mid = gdf.geometry.interpolate(0.5, normalized=True)
    gdf["x_m"], gdf["y_m"] = mid.x.to_numpy(), mid.y.to_numpy()
    gdf["y"] = gdf["estado_cierre"].isin(cfg.positive_states).astype(np.int64)
    gdf["is_condominio"] = (gdf["estado_cierre"] == "condominio").astype(np.int64)


def extract_network_metrics(gdf: gpd.GeoDataFrame) -> tuple[nx.Graph, pd.DataFrame]:
    """Primal graph + per-segment topological metrics (degree, betweenness, closeness,
    clustering, bridge, articulation point, cul-de-sac, endpoint degrees)."""
    g = build_primal_graph(gdf)
    nf = network_features(gdf, g)
    gf = geometry_features(gdf)
    metrics = pd.concat([nf, gf], axis=1)
    assert len(metrics) == len(gdf) and np.isfinite(metrics.to_numpy(float)).all()
    logger.info("Primal graph: %d intersections, %d segments; %d geometric/topological features",
                g.number_of_nodes(), g.number_of_edges(), metrics.shape[1])
    return g, metrics


def build_urban_graph(data_cfg: DataConfig, graph_cfg: GraphConfig) -> UrbanGraph:
    """Full Phase-1 construction honouring ``data_cfg.feature_set``."""
    gdf = load_geometries(data_cfg)
    g, metrics = extract_network_metrics(gdf)
    if data_cfg.feature_set == "geometry_socio":
        gdf = gpd.GeoDataFrame(attach_covariates(pd.DataFrame(gdf.drop(columns="geometry")), data_cfg),
                               geometry=gdf.geometry.values, crs=gdf.crs)
        X, names = assemble_feature_matrix(gdf, [metrics])
    elif data_cfg.feature_set == "geometry":
        socio = [c for c in gdf.columns if c.startswith(SOCIO_PREFIXES)]
        X, names = assemble_feature_matrix(gdf.drop(columns=socio), [metrics])
        assert not any(n.startswith(SOCIO_PREFIXES) for n in names), "socio features leaked into Exp. A"
    else:
        raise ValueError(f"unknown feature_set {data_cfg.feature_set!r}")
    data = build_pyg_data(pd.DataFrame(gdf.drop(columns="geometry")), X, graph_cfg, names)
    assert data.x.shape == (len(gdf), len(names))
    return UrbanGraph(segments=gdf, primal=g, data=data, feature_names=names,
                      feature_set=data_cfg.feature_set)
