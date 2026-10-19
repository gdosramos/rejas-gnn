"""Spatial block cross-validation with buffer zones.

Random node splits are invalid for spatial data: a test segment would have its immediate
neighbours in the training set, and both the GNN's message passing and spatial
autocorrelation of the label would inflate the score. We therefore:

1. partition the study area into ``n_blocks`` contiguous spatial blocks (k-means on metric
   coordinates, seeded);
2. assign blocks to ``n_folds`` folds;
3. for each fold, hold out its blocks as **test**, a fraction of the remaining blocks as
   **validation** (early stopping / threshold choice) and the rest as **train**;
4. remove from train/val every node closer than ``buffer_m`` to any test node (buffer zone),
   so test nodes receive no label information through adjacency or spatial correlation.

Message passing still operates on the full graph (transductive setting), but test labels are
never used in the loss, and buffer nodes carry no supervision at all.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from rejas_gnn.config import SplitConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldMasks:
    """Boolean node masks for a single fold."""

    fold: int
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    buffer: torch.Tensor

    def check(self) -> None:
        """Disjointness and non-emptiness invariants."""
        assert not (self.train & self.val).any(), "train/val overlap"
        assert not (self.train & self.test).any(), "train/test overlap"
        assert not (self.val & self.test).any(), "val/test overlap"
        assert not (self.buffer & self.train).any(), "buffer nodes must not be trained on"
        assert not (self.buffer & self.val).any(), "buffer nodes must not be validated on"
        assert self.train.any() and self.test.any(), "empty train or test"


def spatial_blocks(xy: np.ndarray, n_blocks: int, seed: int) -> np.ndarray:
    """Assign each node to one of ``n_blocks`` contiguous spatial blocks (k-means, seeded)."""
    assert xy.ndim == 2 and xy.shape[1] == 2
    assert 1 <= n_blocks <= len(xy)
    km = KMeans(n_clusters=n_blocks, n_init=10, random_state=seed).fit(xy)
    labels = km.labels_.astype(np.int64)
    assert len(np.unique(labels)) == n_blocks, "k-means produced an empty block"
    return labels


def spatial_block_kfold(
    xy: np.ndarray, cfg: SplitConfig, seed: int, labels: np.ndarray | None = None
) -> list[FoldMasks]:
    """Create ``cfg.n_folds`` spatial folds with validation blocks and buffer zones.

    Parameters
    ----------
    xy:
        ``(N, 2)`` metric coordinates.
    cfg:
        Split configuration.
    seed:
        Controls block construction and fold assignment.
    labels:
        Optional binary labels, only used to guarantee every test fold has >= 1 positive;
        if impossible, a warning is logged.
    """
    n = len(xy)
    blocks = spatial_blocks(xy, cfg.n_blocks, seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(cfg.n_blocks)
    block_to_fold = np.empty(cfg.n_blocks, dtype=np.int64)
    block_to_fold[order] = np.arange(cfg.n_blocks) % cfg.n_folds
    node_fold = block_to_fold[blocks]
    tree = cKDTree(xy)

    folds: list[FoldMasks] = []
    for f in range(cfg.n_folds):
        test = node_fold == f
        if labels is not None and labels[test].sum() == 0:
            logger.warning("Fold %d has no positive test node", f)
        # buffer: non-test nodes within buffer_m of any test node
        near = tree.query_ball_point(xy[test], r=cfg.buffer_m)
        buf_idx = np.unique(np.concatenate([np.asarray(i, dtype=np.int64) for i in near]))
        buffer = np.zeros(n, dtype=bool)
        buffer[buf_idx] = True
        buffer &= ~test
        # validation blocks among the remaining
        remaining_blocks = np.array([b for b in range(cfg.n_blocks) if block_to_fold[b] != f])
        n_val = int(round(cfg.val_fraction_of_blocks * len(remaining_blocks)))
        val_blocks = (rng.choice(remaining_blocks, size=n_val, replace=False) if n_val
                      else np.array([], dtype=np.int64))
        val = np.isin(blocks, val_blocks) & ~test & ~buffer
        train = ~test & ~buffer & ~val
        fm = FoldMasks(
            fold=f,
            train=torch.from_numpy(train),
            val=torch.from_numpy(val),
            test=torch.from_numpy(test),
            buffer=torch.from_numpy(buffer),
        )
        fm.check()
        logger.info(
            "Fold %d: train=%d val=%d test=%d buffer=%d | test positives=%s",
            f, int(train.sum()), int(val.sum()), int(test.sum()), int(buffer.sum()),
            None if labels is None else int(labels[test].sum()),
        )
        folds.append(fm)

    covered = np.zeros(n, dtype=bool)
    for fm in folds:
        covered |= fm.test.numpy()
    assert covered.all(), "every node must be a test node exactly once"
    return folds
