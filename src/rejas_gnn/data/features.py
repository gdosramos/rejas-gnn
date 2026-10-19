"""Node feature engineering for street segments.

Features are computed from the *primal* street graph (OSM nodes) with NetworkX and from
segment geometry. None of them uses the closure label. External covariates (census, MBHT)
are optional and appended by :func:`rejas_gnn.data.load.attach_covariates`.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

from rejas_gnn.data.load import LABEL_LEAK_COLUMNS

logger = logging.getLogger(__name__)

ROAD_HIERARCHY: dict[str, int] = {
    "motorway": 8, "trunk": 7, "primary": 6, "secondary": 5, "tertiary": 4, "unclassified": 3,
    "residential": 2, "living_street": 1, "service": 1, "pedestrian": 0, "footway": 0,
    "path": 0, "cycleway": 0, "busway": 5, "track": 1, "primary_link": 5,
    "secondary_link": 4, "tertiary_link": 3, "trunk_link": 6, "motorway_link": 7,
}
ROAD_GROUPS: tuple[str, ...] = (
    "residential", "living_street", "tertiary", "secondary", "primary", "footway",
    "service", "pedestrian", "trunk", "other",
)
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "nodo_origen", "nodo_destino", "lat_origen", "lon_origen", "lat_destino", "lon_destino",
        "nombre_calle", "tipo_via_osm", "tipo_via_origen", "tipo_via_destino", "año", "comuna",
        "arista_id", "lat_mid", "lon_mid", "x_m", "y_m", "x_utm", "y_utm", "is_condominio",
    }
    | LABEL_LEAK_COLUMNS
)


def build_primal_graph(df: pd.DataFrame) -> nx.Graph:
    """Undirected primal street graph: OSM nodes as vertices, segments as edges.

    Parallel segments between the same pair collapse into one edge (``nx.Graph``), matching
    the original study's ``DiGraph.to_undirected`` behaviour.
    """
    g = nx.Graph()
    for r in df.itertuples(index=False):
        g.add_node(int(r.nodo_origen), x=float(r.lon_origen), y=float(r.lat_origen))
        g.add_node(int(r.nodo_destino), x=float(r.lon_destino), y=float(r.lat_destino))
        g.add_edge(int(r.nodo_origen), int(r.nodo_destino), length=float(r.distancia_m))
    assert g.number_of_nodes() > 0
    return g


def network_features(df: pd.DataFrame, g: nx.Graph, betweenness_k: int | None = None,
                     seed: int = 0) -> pd.DataFrame:
    """Per-segment network metrics derived from endpoint statistics of the primal graph.

    Parameters
    ----------
    betweenness_k:
        If given, approximate betweenness with ``k`` sampled sources (large graphs).
    """
    n = g.number_of_nodes()
    k = None if betweenness_k is None else min(betweenness_k, n)
    deg_c = nx.degree_centrality(g)
    bet_c = nx.betweenness_centrality(g, k=k, weight="length", seed=seed)
    clo_c = nx.closeness_centrality(g, distance="length") if n < 5000 else deg_c
    clust = nx.clustering(g)
    bridges = set(nx.bridges(g))
    artic = set(nx.articulation_points(g))
    deg = dict(g.degree())

    o = df["nodo_origen"].astype(int)
    d = df["nodo_destino"].astype(int)
    out = pd.DataFrame(index=df.index)
    for name, dic in (("degree_c", deg_c), ("betweenness", bet_c), ("closeness", clo_c),
                      ("clustering", clust)):
        a, b = o.map(dic).astype(float), d.map(dic).astype(float)
        out[f"{name}_avg"] = (a + b) / 2.0
        out[f"{name}_max"] = np.maximum(a, b)
    go, gd = o.map(deg).astype(float), d.map(deg).astype(float)
    out["grado_min"] = np.minimum(go, gd)
    out["grado_max"] = np.maximum(go, gd)
    out["es_puente"] = [float((a, b) in bridges or (b, a) in bridges) for a, b in zip(o, d, strict=False)]
    out["tiene_articulacion"] = [float(a in artic or b in artic) for a, b in zip(o, d, strict=False)]
    out["es_culdesac"] = ((go == 1) | (gd == 1)).astype(float)
    assert not out.isna().any().any(), "NaN in network features"
    return out


def geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Length, orientation, distance to centroid and road hierarchy / one-hot road group."""
    out = pd.DataFrame(index=df.index)
    out["log_length"] = np.log1p(df["distancia_m"].astype(float))
    ang = np.degrees(np.arctan2(df["lat_destino"] - df["lat_origen"],
                                df["lon_destino"] - df["lon_origen"])) % 180.0
    out["orient_sin"] = np.sin(np.radians(2 * ang))
    out["orient_cos"] = np.cos(np.radians(2 * ang))
    cx, cy = df["x_m"].mean(), df["y_m"].mean()
    out["dist_centroid_km"] = np.hypot(df["x_m"] - cx, df["y_m"] - cy) / 1000.0
    out["road_hierarchy"] = df["tipo_via_osm"].map(ROAD_HIERARCHY).fillna(2).astype(float)
    grp = df["tipo_via_osm"].where(df["tipo_via_osm"].isin(ROAD_GROUPS[:-1]), "other")
    for g_ in ROAD_GROUPS:
        out[f"via_{g_}"] = (grp == g_).astype(float)
    return out


def assemble_feature_matrix(df: pd.DataFrame, extra: list[pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
    """Concatenate feature blocks + external covariates present in ``df``; return (X, names).

    NaNs in external covariates are median-imputed (logged) and an indicator column is added
    per column with missing values so the model can learn from missingness.
    """
    blocks = list(extra)
    ext_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS
                and c.startswith(("cpv24_", "uai_"))]
    if ext_cols:
        ext = df[ext_cols].apply(pd.to_numeric, errors="coerce")
        na_cols = [c for c in ext_cols if ext[c].isna().any()]
        indicators = {f"{c}__missing": ext[c].isna().astype(float) for c in na_cols}
        ext = ext.fillna(ext.median(numeric_only=True))
        ext = pd.concat([ext, pd.DataFrame(indicators, index=ext.index)], axis=1)
        if na_cols:
            logger.info("Median-imputed %d covariate columns with NaN", len(na_cols))
        blocks.append(ext)
    feat = pd.concat(blocks, axis=1)
    feat = feat.loc[:, feat.std(ddof=0) > 0]  # drop constant columns
    leaked = LABEL_LEAK_COLUMNS.intersection(feat.columns)
    assert not leaked, f"label leakage into features: {sorted(leaked)}"
    X = feat.to_numpy(dtype=np.float32)
    assert np.isfinite(X).all(), "non-finite feature values"
    logger.info("Feature matrix: %d nodes x %d features", *X.shape)
    return X, list(feat.columns)
