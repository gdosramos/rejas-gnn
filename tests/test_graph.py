"""Graph construction invariants."""
import numpy as np
import pytest
import torch
from torch_geometric.utils import contains_self_loops, is_undirected

from rejas_gnn.config import GraphConfig
from rejas_gnn.data.graph import (
    build_pyg_data,
    line_graph_edges,
    radius_graph_edges,
    validate_graph,
)


def test_line_graph_matches_shared_endpoints(toy_segments):
    ei = line_graph_edges(toy_segments)
    pairs = {tuple(sorted(p)) for p in ei.T.tolist()}
    assert pairs == {(0, 1), (1, 2)}
    assert 3 not in ei  # isolated segment


def test_line_graph_symmetric_no_self_loops(toy_segments):
    ei = torch.from_numpy(line_graph_edges(toy_segments))
    assert is_undirected(ei, num_nodes=4)
    assert not contains_self_loops(ei)


def test_radius_graph_respects_radius():
    xy = np.array([[0, 0], [50, 0], [500, 0], [520, 0]], dtype=float)
    ei = radius_graph_edges(xy, radius_m=100.0, max_neighbors=5)
    pairs = {tuple(sorted(p)) for p in ei.T.tolist()}
    assert pairs == {(0, 1), (2, 3)}


def test_radius_graph_max_neighbors_bound():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 10, size=(200, 2))  # dense cluster
    ei = radius_graph_edges(xy, radius_m=100.0, max_neighbors=3)
    out_deg = np.bincount(ei[0], minlength=200)
    assert out_deg.max() <= 3


@pytest.mark.parametrize("gtype", ["line", "radius", "union"])
def test_build_pyg_data_valid(toy_segments, gtype):
    X = np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)
    data = build_pyg_data(toy_segments, X, GraphConfig(type=gtype, radius_m=150.0, max_neighbors=4))
    validate_graph(data)
    assert data.num_nodes == 4 and data.x.shape == (4, 3)
    assert data.y.tolist() == [0, 1, 0, 0]


def test_union_is_superset(toy_segments):
    X = np.zeros((4, 2), dtype=np.float32) + np.arange(4)[:, None]
    line = build_pyg_data(toy_segments, X, GraphConfig(type="line"))
    union = build_pyg_data(toy_segments, X, GraphConfig(type="union", radius_m=150.0))
    s_line = {tuple(p) for p in line.edge_index.T.tolist()}
    s_union = {tuple(p) for p in union.edge_index.T.tolist()}
    assert s_line <= s_union


def test_validate_rejects_self_loops(toy_segments):
    X = np.zeros((4, 2), dtype=np.float32)
    data = build_pyg_data(toy_segments, X, GraphConfig(type="line"))
    data.edge_index = torch.tensor([[0, 0], [0, 0]])
    with pytest.raises(AssertionError):
        validate_graph(data)
