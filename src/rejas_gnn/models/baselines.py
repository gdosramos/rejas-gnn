"""Non-relational baselines that ignore ``edge_index``.

They answer the key scientific question: *does the graph add predictive information beyond
the node's own attributes?* A GNN that does not beat them has learned nothing relational.
"""
from __future__ import annotations

from torch import Tensor, nn


class LogisticRegression(nn.Module):
    """Linear logit (equivalent to L2-regularised logistic regression via weight decay)."""

    def __init__(self, in_dim: int, **_: object) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        return self.lin(x).squeeze(-1)


class MLP(nn.Module):
    """Two-hidden-layer perceptron with LayerNorm and dropout."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, **_: object) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        return self.head(self.body(x)).squeeze(-1)
