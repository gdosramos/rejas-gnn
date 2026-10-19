"""Model factory keyed by ``ModelConfig.name``."""
from __future__ import annotations

from torch import nn

from rejas_gnn.config import ModelConfig
from rejas_gnn.models.baselines import MLP, LogisticRegression
from rejas_gnn.models.gnn_architectures import GAT, GCN, GraphSAGE

_REGISTRY: dict[str, type[nn.Module]] = {
    "logreg": LogisticRegression,
    "mlp": MLP,
    "gcn": GCN,
    "sage": GraphSAGE,
    "gat": GAT,
}

USES_GRAPH: frozenset[str] = frozenset({"gcn", "sage", "gat"})


def build_model(cfg: ModelConfig, in_dim: int) -> nn.Module:
    """Instantiate the model named in ``cfg`` for ``in_dim`` input features."""
    if cfg.name not in _REGISTRY:
        raise KeyError(f"Unknown model {cfg.name!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[cfg.name](
        in_dim=in_dim, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        dropout=cfg.dropout, heads=cfg.heads, aggr=cfg.aggr,
    )
