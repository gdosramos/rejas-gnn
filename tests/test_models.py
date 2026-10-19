"""Model forward shapes, graph usage and determinism."""
import pytest
import torch

from rejas_gnn.config import ModelConfig
from rejas_gnn.models.registry import USES_GRAPH, build_model
from rejas_gnn.seed import set_global_seed


@pytest.mark.parametrize("name", ["logreg", "mlp", "gcn", "sage", "gat"])
def test_forward_shape(random_graph, name):
    set_global_seed(0)
    m = build_model(ModelConfig(name=name, hidden_dim=16, heads=4), in_dim=6)
    out = m(random_graph.x, random_graph.edge_index)
    assert out.shape == (400,) and torch.isfinite(out).all()


@pytest.mark.parametrize("name", sorted(USES_GRAPH))
def test_gnn_depends_on_edges(random_graph, name):
    set_global_seed(0)
    m = build_model(ModelConfig(name=name, hidden_dim=16, dropout=0.0), in_dim=6).eval()
    with torch.no_grad():
        a = m(random_graph.x, random_graph.edge_index)
        b = m(random_graph.x, torch.empty(2, 0, dtype=torch.long))
    assert not torch.allclose(a, b), f"{name} ignores the graph"


def test_baselines_ignore_edges(random_graph):
    set_global_seed(0)
    m = build_model(ModelConfig(name="mlp", hidden_dim=16, dropout=0.0), in_dim=6).eval()
    with torch.no_grad():
        a = m(random_graph.x, random_graph.edge_index)
        b = m(random_graph.x, torch.empty(2, 0, dtype=torch.long))
    assert torch.allclose(a, b)


def test_seed_reproducibility(random_graph):
    outs = []
    for _ in range(2):
        set_global_seed(123)
        m = build_model(ModelConfig(name="gat", hidden_dim=16), in_dim=6).eval()
        with torch.no_grad():
            outs.append(m(random_graph.x, random_graph.edge_index))
    assert torch.equal(outs[0], outs[1])


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model(ModelConfig(name="transformer"), in_dim=3)  # type: ignore[arg-type]


def test_gat_attention_export(random_graph):
    set_global_seed(0)
    m = build_model(ModelConfig(name="gat", hidden_dim=16, heads=4), in_dim=6)
    ei, alpha = m.attention(random_graph.x, random_graph.edge_index)
    assert alpha.shape == (ei.size(1), 4) and torch.isfinite(alpha).all()
