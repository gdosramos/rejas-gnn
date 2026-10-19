"""End-to-end pipeline: raw CSV -> features -> PyG graph -> spatial CV -> results."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import Data

from rejas_gnn.config import ExperimentConfig
from rejas_gnn.data.graph_builder import build_urban_graph
from rejas_gnn.logging_utils import dump_json, environment_fingerprint
from rejas_gnn.seed import set_global_seed
from rejas_gnn.train import run_experiment

logger = logging.getLogger(__name__)


def build_dataset(cfg: ExperimentConfig, cache: Path | None = None) -> tuple[Data, pd.DataFrame]:
    """Build (or load from ``cache``) the PyG dataset for the configured graph type."""
    if cache is not None and cache.exists():
        logger.info("Loading cached dataset %s", cache)
        payload = torch.load(cache, weights_only=False)
        return payload["data"], payload["df"]
    set_global_seed(cfg.train.seeds[0])
    ug = build_urban_graph(cfg.data, cfg.graph)
    data, df = ug.data, pd.DataFrame(ug.segments.drop(columns="geometry"))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"data": data, "df": df}, cache)
        logger.info("Cached dataset to %s", cache)
    return data, df


def run(cfg: ExperimentConfig) -> dict[str, Any]:
    """Run one configured experiment and persist results under ``results/<name>/``."""
    out_dir = Path(cfg.results_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path("data/processed") / f"dataset_{cfg.graph.type}_{cfg.data.feature_set}_{Path(cfg.data.raw_edges_csv).stem}.pt"
    data, df = build_dataset(cfg, cache=cache)
    res = run_experiment(data, cfg)

    preds = pd.DataFrame({
        "arista_id": data.arista_id, "x_m": df["x_m"], "y_m": df["y_m"], "y": data.y.numpy(),
        "estado_cierre": df["estado_cierre"], "p_oof": res.pop("oof_prob_mean"),
    })
    preds.to_csv(out_dir / "predictions_oof.csv", index=False)
    pd.DataFrame(res["per_fold"]).to_csv(out_dir / "metrics_per_fold.csv", index=False)
    dump_json({"config": cfg.to_dict(), "environment": environment_fingerprint(),
               "n_nodes": data.num_nodes, "n_edges": int(data.edge_index.size(1)),
               "n_features": int(data.x.size(1)), "feature_names": data.feature_names,
               **{k: v for k, v in res.items() if k != "per_fold"}}, out_dir / "summary.json")
    a = res["aggregate"]
    logger.info("== %s | %s graph | PR-AUC %.3f±%.3f | ROC-AUC %.3f±%.3f | F1 %.3f±%.3f | resid Moran %.3f",
                cfg.model.name, cfg.graph.type, a["pr_auc"]["mean"], a["pr_auc"]["std"],
                a["roc_auc"]["mean"], a["roc_auc"]["std"], a["f1"]["mean"], a["f1"]["std"],
                res["residual_morans_i"])
    return res
