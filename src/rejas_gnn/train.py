"""Full-batch training loop with early stopping and spatial cross-validation."""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from rejas_gnn.config import ExperimentConfig
from rejas_gnn.data.splits import FoldMasks, spatial_block_kfold
from rejas_gnn.metrics import best_f1_threshold, classification_metrics, morans_i
from rejas_gnn.models.registry import USES_GRAPH, build_model
from rejas_gnn.seed import set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Outcome of one (seed, fold) run."""

    seed: int
    fold: int
    best_epoch: int
    threshold: float
    val: dict[str, float]
    test: dict[str, float]
    test_prob: np.ndarray = field(repr=False)
    test_index: np.ndarray = field(repr=False)
    val_prob: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    val_index: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), repr=False)


def _pos_weight(y: Tensor, mask: Tensor, setting: str | float) -> Tensor:
    pos = float(y[mask].sum())
    neg = float(mask.sum()) - pos
    assert pos > 0, "no positive training node"
    w = neg / pos if setting == "balanced" else float(setting)
    return torch.tensor([w], dtype=torch.float32)


def standardise(x: Tensor, train_mask: Tensor) -> Tensor:
    """Z-score features using **training nodes only** (no test statistics leak)."""
    mu = x[train_mask].mean(0, keepdim=True)
    sd = x[train_mask].std(0, keepdim=True).clamp_min(1e-6)
    out = (x - mu) / sd
    assert torch.isfinite(out).all()
    return out


def train_one_fold(data: Data, masks: FoldMasks, cfg: ExperimentConfig, seed: int) -> FoldResult:
    """Train on ``masks.train``, early-stop on validation PR-AUC, evaluate on test.

    If the validation mask is empty (``val_fraction_of_blocks = 0``) early stopping falls
    back to training loss, which is logged as a warning.
    """
    set_global_seed(seed)
    device = torch.device(cfg.train.device)
    x = standardise(data.x, masks.train).to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    train_m, val_m, test_m = (m.to(device) for m in (masks.train, masks.val, masks.test))

    model = build_model(cfg.model, in_dim=x.size(1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y, train_m, cfg.train.pos_weight).to(device))
    use_val = bool(val_m.any())
    if not use_val:
        logger.warning("Empty validation set: early stopping on training loss")

    best_score, best_state, best_epoch, bad = -np.inf, copy.deepcopy(model.state_dict()), 0, 0
    yf = y.float()
    t0 = time.time()
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, edge_index)
        loss = loss_fn(logits[train_m], yf[train_m])
        assert torch.isfinite(loss), f"non-finite loss at epoch {epoch}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(x, edge_index)).cpu().numpy()
        if use_val:
            yv = y[val_m].cpu().numpy()
            score = classification_metrics(yv, p[val_m.cpu().numpy()], 0.5)["pr_auc"]
        else:
            score = -float(loss)
        if score > best_score + 1e-6:
            best_score, best_state, best_epoch, bad = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
        if epoch % 50 == 0 or epoch == 1:
            logger.info("seed=%d fold=%d epoch=%d loss=%.4f val_pr_auc=%.4f",
                        seed, masks.fold, epoch, loss.item(), score)
        if bad >= cfg.train.patience and epoch >= cfg.train.min_epochs:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(x, edge_index)).cpu().numpy()
    yn = y.cpu().numpy()
    vm, tm = val_m.cpu().numpy(), test_m.cpu().numpy()
    thr = best_f1_threshold(yn[vm], p[vm]) if use_val else 0.5
    val_metrics = classification_metrics(yn[vm], p[vm], thr) if use_val else {}
    test_metrics = classification_metrics(yn[tm], p[tm], thr)
    logger.info(
        "seed=%d fold=%d done in %.1fs | best_epoch=%d | test PR-AUC=%.3f ROC-AUC=%.3f F1=%.3f",
        seed, masks.fold, time.time() - t0, best_epoch, test_metrics["pr_auc"],
        test_metrics["roc_auc"], test_metrics["f1"],
    )
    return FoldResult(seed, masks.fold, best_epoch, thr, val_metrics, test_metrics, p[tm], np.flatnonzero(tm),
                      p[vm], np.flatnonzero(vm))


def run_experiment(data: Data, cfg: ExperimentConfig) -> dict[str, Any]:
    """Repeated spatial-CV over all seeds; returns aggregated metrics and OOF predictions."""
    if cfg.model.name not in USES_GRAPH:
        logger.info("Model %s ignores the graph (non-relational baseline)", cfg.model.name)
    xy = data.pos.numpy().astype(np.float64)
    y = data.y.numpy()
    label_moran = morans_i(y.astype(float), data.edge_index, data.num_nodes)
    logger.info("Moran's I of the label on the graph: %.3f", label_moran)

    results: list[FoldResult] = []
    oof = np.full((len(cfg.train.seeds), data.num_nodes), np.nan)
    for si, seed in enumerate(cfg.train.seeds):
        folds = spatial_block_kfold(xy, cfg.split, seed=seed, labels=y)
        for fm in folds:
            r = train_one_fold(data, fm, cfg, seed)
            results.append(r)
            oof[si, r.test_index] = r.test_prob
    assert np.isfinite(oof).all(), "every node must receive an out-of-fold prediction per seed"

    keys = ["pr_auc", "roc_auc", "f1", "precision", "recall", "mcc", "brier"]
    per_fold = [{"seed": r.seed, "fold": r.fold, "best_epoch": r.best_epoch, **r.test} for r in results]
    agg = {k: {"mean": float(np.nanmean([r.test[k] for r in results])),
               "std": float(np.nanstd([r.test[k] for r in results], ddof=1)) if len(results) > 1 else 0.0}
           for k in keys}
    oof_mean = oof.mean(0)
    resid_moran = morans_i(y - oof_mean, data.edge_index, data.num_nodes)
    oof_metrics = classification_metrics(y, oof_mean, float(np.median([r.threshold for r in results])))
    return {
        "model": cfg.model.name,
        "graph": cfg.graph.type,
        "aggregate": agg,
        "oof_pooled": oof_metrics,
        "label_morans_i": label_moran,
        "residual_morans_i": resid_moran,
        "per_fold": per_fold,
        "oof_prob_mean": oof_mean,
    }
