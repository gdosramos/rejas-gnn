"""GNN architectures for node (segment) classification.

All models output a single logit per node and share the same interface:
``forward(x, edge_index) -> Tensor[N]``.

Design rationale
----------------
* **GraphSAGE** (Hamilton et al., 2017) is the primary model. Its update
  ``h_v' = W_1 h_v + W_2 · mean_{u∈N(v)} h_u`` keeps separate weights for the node itself
  and for its neighbourhood, which matters here: closures are a *local* decision (own road
  type, dead-end status) modulated by context (closures nearby). Mean aggregation is
  robust to the heavy-tailed degree of line graphs, and the model is *inductive*: it can be
  applied unchanged to another comuna (e.g. La Florida) without retraining.
* **GCN** (Kipf & Welling, 2017) is the classical symmetric-normalised baseline; it mixes
  self and neighbour information with the same weights, which tends to over-smooth when the
  label is heterophilous (open segments next to closed ones).
* **GATv2** (Brody et al., 2022) learns per-edge attention; attention weights are exported
  for interpretability (which neighbours drive a prediction).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv


class _GNNBase(nn.Module):
    """Shared scaffolding: input projection, ``num_layers`` conv blocks, output head."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        assert in_dim > 0 and hidden_dim > 0 and num_layers >= 1
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.head = nn.Linear(hidden_dim, 1)

    def _block(self, i: int, h: Tensor, edge_index: Tensor) -> Tensor:
        h = self.convs[i](h, edge_index)
        h = self.norms[i](h)
        h = F.relu(h)
        return F.dropout(h, p=self.dropout, training=self.training)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return one logit per node (shape ``[N]``)."""
        assert x.dim() == 2, "x must be [N, F]"
        h = x
        for i in range(len(self.convs)):
            h = self._block(i, h, edge_index)
        return self.head(h).squeeze(-1)


class GCN(_GNNBase):
    """Graph Convolutional Network (Kipf & Welling, 2017)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, **_: object) -> None:
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        dims = [in_dim] + [hidden_dim] * num_layers
        for a, b in zip(dims[:-1], dims[1:], strict=False):
            self.convs.append(GCNConv(a, b, add_self_loops=True, normalize=True))
            self.norms.append(nn.LayerNorm(b))


class GraphSAGE(_GNNBase):
    """GraphSAGE (Hamilton et al., 2017) with configurable aggregation."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, aggr: str = "mean", **_: object) -> None:
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        dims = [in_dim] + [hidden_dim] * num_layers
        for a, b in zip(dims[:-1], dims[1:], strict=False):
            self.convs.append(SAGEConv(a, b, aggr=aggr, root_weight=True))
            self.norms.append(nn.LayerNorm(b))


class GAT(_GNNBase):
    """GATv2 (Brody et al., 2022); multi-head with concatenation in hidden layers."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, heads: int = 4, **_: object) -> None:
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        per_head = hidden_dim // heads
        dim_in = in_dim
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(dim_in, per_head, heads=heads, concat=True, dropout=dropout,
                          add_self_loops=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            dim_in = hidden_dim

    @torch.no_grad()
    def attention(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(edge_index_with_self_loops, alpha)`` of the last layer for interpretation."""
        self.eval()
        h = x
        for i in range(len(self.convs) - 1):
            h = self._block(i, h, edge_index)
        _, (ei, alpha) = self.convs[-1](h, edge_index, return_attention_weights=True)
        return ei, alpha
