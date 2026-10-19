"""Graph construction over street segments (nodes = OSM edges).

Two topologies are supported and can be combined:

* ``line``   - the *line graph* of the primal street network: two segments are adjacent iff
  they share an OSM node. Encodes walkable connectivity (a gate on one segment affects the
  through-traffic of its neighbours).
* ``radius`` - segments whose midpoints lie within ``radius_m`` are adjacent (Euclidean
  proximity). Encodes neighbourhood context independent of connectivity.
* ``union``  - union of both edge sets.

All graphs are undirected, without self-loops, and returned as a PyG ``Data`` object.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, contains_self_loops, is_undirected, to_undirected

from rejas_gnn.config import GraphConfig

logger = logging.getLogger(__name__)


def line_graph_edges(df: pd.DataFrame) -> np.ndarray:
    """Edge list ``(2, E)`` of the line graph: segments sharing at least one OSM node.

    Implemented through the sparse incidence matrix ``B`` (segments x OSM nodes):
    ``A = B B^T`` has a non-zero off-diagonal entry for every pair of segments that share an
    endpoint. Self-loops (the diagonal) are removed. Complexity is linear in the sum of
    squared OSM-node degrees, i.e. negligible for street networks.
    """
    n = len(df)
    nodes = pd.concat([df["nodo_origen"], df["nodo_destino"]]).astype(np.int64)
    node_index = {v: i for i, v in enumerate(np.unique(nodes.to_numpy()))}
    rows = np.repeat(np.arange(n), 2)
    cols = np.array([node_index[v] for v in nodes.to_numpy()])
    cols = cols.reshape(2, n).T.ravel()  # realign to (seg0_o, seg0_d, seg1_o, ...)
    B = sp.csr_matrix((np.ones(2 * n), (rows, cols)), shape=(n, len(node_index)))
    A = (B @ B.T).tocoo()
    mask = A.row != A.col
    edges = np.vstack([A.row[mask], A.col[mask]]).astype(np.int64)
    return edges


def radius_graph_edges(xy: np.ndarray, radius_m: float, max_neighbors: int) -> np.ndarray:
    """Edge list ``(2, E)`` connecting segment midpoints within ``radius_m``.

    Each node keeps at most ``max_neighbors`` nearest neighbours inside the radius, which
    bounds the degree in dense areas. The result is symmetrised by the caller.
    """
    assert xy.ndim == 2 and xy.shape[1] == 2, "xy must be (N, 2)"
    assert radius_m > 0 and max_neighbors >= 1
    tree = cKDTree(xy)
    dist, idx = tree.query(xy, k=max_neighbors + 1, distance_upper_bound=radius_m)
    src = np.repeat(np.arange(len(xy)), max_neighbors + 1)
    dst = idx.ravel()
    ok = np.isfinite(dist.ravel()) & (dst != src) & (dst < len(xy))
    return np.vstack([src[ok], dst[ok]]).astype(np.int64)


def build_pyg_data(
    df: pd.DataFrame, X: np.ndarray, cfg: GraphConfig, feature_names: list[str] | None = None
) -> Data:
    """Assemble a validated, undirected PyG ``Data`` object for node classification.

    Attributes: ``x`` (features), ``y`` (int64 labels), ``pos`` (metric coordinates),
    ``edge_index``, ``arista_id`` (list[str]), ``feature_names``.
    """
    n = len(df)
    assert X.shape[0] == n, "feature/segment count mismatch"
    xy = df[["x_m", "y_m"]].to_numpy(dtype=np.float64)

    parts: list[np.ndarray] = []
    if cfg.type in ("line", "union"):
        parts.append(line_graph_edges(df))
    if cfg.type in ("radius", "union"):
        parts.append(radius_graph_edges(xy, cfg.radius_m, cfg.max_neighbors))
    if not parts:
        raise ValueError(f"Unknown graph type {cfg.type!r}")
    raw = np.concatenate(parts, axis=1)
    edge_index = torch.from_numpy(raw).long()
    edge_index = to_undirected(edge_index, num_nodes=n)
    edge_index = coalesce(edge_index, num_nodes=n)

    data = Data(
        x=torch.from_numpy(X).float(),
        y=torch.from_numpy(df["y"].to_numpy(dtype=np.int64).copy()),
        pos=torch.from_numpy(xy.copy()).float(),
        edge_index=edge_index,
    )
    data.arista_id = df["arista_id"].tolist()
    data.feature_names = feature_names or []
    validate_graph(data)
    deg = torch.bincount(edge_index[0], minlength=n).float()
    logger.info(
        "Graph[%s]: %d nodes, %d directed edges, mean degree %.2f, isolated %d",
        cfg.type, n, edge_index.size(1), deg.mean().item(), int((deg == 0).sum()),
    )
    return data


def validate_graph(data: Data) -> None:
    """Defensive checks that every downstream module relies on."""
    n = data.num_nodes
    assert n is not None and n > 0, "empty graph"
    assert data.edge_index.dtype == torch.long
    assert data.edge_index.min() >= 0 and data.edge_index.max() < n, "edge index out of range"
    assert is_undirected(data.edge_index, num_nodes=n), "graph must be undirected"
    assert not contains_self_loops(data.edge_index), "self-loops are not allowed"
    assert data.x.size(0) == n and data.y.size(0) == n and data.pos.size(0) == n
    assert torch.isfinite(data.x).all(), "non-finite node features"
    assert set(data.y.unique().tolist()) <= {0, 1}, "labels must be binary"
