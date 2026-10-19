"""Shared fixtures: a tiny synthetic street network with known topology."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from rejas_gnn.config import ExperimentConfig


@pytest.fixture
def toy_segments() -> pd.DataFrame:
    """Four segments on a path 1-2-3-4 plus an isolated segment 5-6.

    Line-graph adjacency expected: (0,1), (1,2) ; segment 3 (5-6) isolated.
    Segment index: 0=(1,2) 1=(2,3) 2=(3,4) 3=(5,6)
    """
    rows = [
        (1, 2, -33.48, -70.68, -33.481, -70.68, "residential", "abierta"),
        (2, 3, -33.481, -70.68, -33.482, -70.68, "living_street", "cierre"),
        (3, 4, -33.482, -70.68, -33.483, -70.68, "residential", "abierta"),
        (5, 6, -33.50, -70.70, -33.501, -70.70, "footway", "condominio"),
    ]
    df = pd.DataFrame(rows, columns=["nodo_origen", "nodo_destino", "lat_origen", "lon_origen",
                                     "lat_destino", "lon_destino", "tipo_via_osm", "estado_cierre"])
    df["distancia_m"] = 110.0
    df["arista_id"] = df.nodo_origen.astype(str) + "_" + df.nodo_destino.astype(str) + "_0"
    df["y"] = (df.estado_cierre == "cierre").astype(int)
    df["x_m"] = np.array([0.0, 100.0, 200.0, 5000.0])
    df["y_m"] = np.array([0.0, 0.0, 0.0, 5000.0])
    return df


@pytest.fixture
def random_graph() -> Data:
    """Random geometric graph with 400 nodes and a spatially autocorrelated binary label."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 2000, size=(400, 2))
    from rejas_gnn.data.graph import radius_graph_edges
    ei = torch.from_numpy(radius_graph_edges(xy, 150.0, 8)).long()
    from torch_geometric.utils import coalesce, to_undirected
    ei = coalesce(to_undirected(ei, num_nodes=400), num_nodes=400)
    x = rng.normal(size=(400, 6)).astype(np.float32)
    y = (xy[:, 0] + rng.normal(scale=300, size=400) > 1400).astype(np.int64)  # ~30% positives, clustered
    d = Data(x=torch.from_numpy(x), y=torch.from_numpy(y), pos=torch.from_numpy(xy).float(), edge_index=ei)
    d.arista_id = [str(i) for i in range(400)]
    d.feature_names = [f"f{i}" for i in range(6)]
    return d


@pytest.fixture
def small_cfg() -> ExperimentConfig:
    from rejas_gnn.config import ModelConfig, SplitConfig, TrainConfig
    return ExperimentConfig(
        name="test", split=SplitConfig(n_blocks=8, n_folds=4, buffer_m=100.0, val_fraction_of_blocks=0.25),
        model=ModelConfig(name="sage", hidden_dim=16, num_layers=2, dropout=0.1),
        train=TrainConfig(epochs=15, patience=5, seeds=(0,)),
    )
