"""Typed experiment configuration loaded from YAML.

Every parameter of the pipeline lives here; notebooks and scripts must not hard-code any.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

GraphType = Literal["line", "radius", "union"]
FeatureSet = Literal["geometry", "geometry_socio"]
ModelName = Literal["logreg", "mlp", "gcn", "sage", "gat"]


@dataclass(frozen=True)
class DataConfig:
    """Input files and label definition."""

    raw_edges_csv: Path = Path("data/raw/pac.csv")
    covariates_csv: Path | None = Path("data/external/aristas_PAC_consolidado.csv")
    covariate_prefixes: tuple[str, ...] = ("cpv24_", "uai_")
    positive_states: tuple[str, ...] = ("cierre", "informal", "medio")
    drop_condominio: bool = False
    feature_set: FeatureSet = "geometry_socio"   # ablation: "geometry" = topology+geometry only
    crs_metric: str = "EPSG:32719"
    crs_geographic: str = "EPSG:4326"


@dataclass(frozen=True)
class GraphConfig:
    """Graph topology over street segments (nodes = OSM edges)."""

    type: GraphType = "line"
    radius_m: float = 150.0
    max_neighbors: int = 32


@dataclass(frozen=True)
class SplitConfig:
    """Spatial block cross-validation with a buffer to prevent spatial leakage."""

    n_blocks: int = 20
    n_folds: int = 5
    buffer_m: float = 200.0
    val_fraction_of_blocks: float = 0.2


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyper-parameters (shared across model families)."""

    name: ModelName = "sage"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    heads: int = 4  # GAT only
    aggr: str = "mean"  # SAGE only


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation and early stopping."""

    epochs: int = 300
    lr: float = 5e-3
    weight_decay: float = 5e-4
    patience: int = 40
    min_epochs: int = 30
    pos_weight: Literal["balanced"] | float = "balanced"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    device: str = "cpu"


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration."""

    name: str = "sage_line_pac"
    results_dir: Path = Path("results")
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (paths as strings)."""
        return asdict(self)


def _build(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = dict(raw or {})
    for key, val in list(raw.items()):
        if key.endswith(("_csv", "_dir")) and val is not None:
            raw[key] = Path(val)
        elif isinstance(val, list):
            raw[key] = tuple(val)
    unknown = set(raw) - set(cls.__dataclass_fields__)
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: Path | str, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from YAML, applying optional ``section.key=value`` overrides.

    Parameters
    ----------
    path:
        YAML file.
    overrides:
        Mapping like ``{"model.name": "gat", "train.epochs": 10}``.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    for dotted, value in (overrides or {}).items():
        if "." not in dotted:  # top-level key such as name / results_dir
            raw[dotted] = value
            continue
        section, key = dotted.split(".", 1)
        raw.setdefault(section, {})[key] = value
    top = {k: v for k, v in raw.items() if k in ("name", "results_dir")}
    if "results_dir" in top:
        top["results_dir"] = Path(top["results_dir"])
    cfg = ExperimentConfig(
        **top,
        data=_build(DataConfig, raw.get("data")),
        graph=_build(GraphConfig, raw.get("graph")),
        split=_build(SplitConfig, raw.get("split")),
        model=_build(ModelConfig, raw.get("model")),
        train=_build(TrainConfig, raw.get("train")),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: ExperimentConfig) -> None:
    assert cfg.graph.radius_m > 0, "graph.radius_m must be > 0"
    assert cfg.split.n_folds >= 2, "split.n_folds must be >= 2"
    assert cfg.split.n_blocks >= cfg.split.n_folds, "n_blocks must be >= n_folds"
    assert 0.0 <= cfg.split.val_fraction_of_blocks < 1.0, "val_fraction_of_blocks in [0,1)"
    assert cfg.split.buffer_m >= 0, "buffer_m must be >= 0"
    assert 0.0 <= cfg.model.dropout < 1.0, "dropout in [0,1)"
    assert cfg.model.num_layers >= 1, "num_layers >= 1"
    assert cfg.train.epochs >= 1 and cfg.train.patience >= 1
    assert 0 <= cfg.train.min_epochs <= cfg.train.epochs, "min_epochs in [0, epochs]"
    assert len(cfg.train.seeds) >= 1, "at least one seed"
