"""Spatial block CV: disjointness, coverage, buffer and determinism."""
import numpy as np
from scipy.spatial import cKDTree

from rejas_gnn.config import SplitConfig
from rejas_gnn.data.splits import spatial_block_kfold


def _xy(n=500, seed=0):
    return np.random.default_rng(seed).uniform(0, 3000, size=(n, 2))


def test_every_node_tested_once():
    cfg = SplitConfig(n_blocks=10, n_folds=5, buffer_m=100.0, val_fraction_of_blocks=0.2)
    folds = spatial_block_kfold(_xy(), cfg, seed=0)
    count = sum(f.test.numpy().astype(int) for f in folds)
    assert (count == 1).all()


def test_buffer_excludes_near_test_nodes():
    xy = _xy()
    cfg = SplitConfig(n_blocks=10, n_folds=5, buffer_m=150.0, val_fraction_of_blocks=0.2)
    for f in spatial_block_kfold(xy, cfg, seed=3):
        tree = cKDTree(xy[f.test.numpy()])
        d, _ = tree.query(xy[f.train.numpy() | f.val.numpy()])
        assert (d > 150.0).all(), "a supervised node lies inside the buffer"


def test_masks_disjoint_and_cover():
    xy = _xy()
    cfg = SplitConfig(n_blocks=12, n_folds=4, buffer_m=50.0, val_fraction_of_blocks=0.25)
    for f in spatial_block_kfold(xy, cfg, seed=1):
        total = f.train.int() + f.val.int() + f.test.int() + f.buffer.int()
        assert (total == 1).all()


def test_deterministic_given_seed():
    xy = _xy()
    cfg = SplitConfig(n_blocks=10, n_folds=5, buffer_m=100.0, val_fraction_of_blocks=0.2)
    a = spatial_block_kfold(xy, cfg, seed=7)
    b = spatial_block_kfold(xy, cfg, seed=7)
    for fa, fb in zip(a, b, strict=False):
        assert (fa.train == fb.train).all() and (fa.test == fb.test).all()


def test_zero_val_fraction():
    cfg = SplitConfig(n_blocks=8, n_folds=4, buffer_m=0.0, val_fraction_of_blocks=0.0)
    for f in spatial_block_kfold(_xy(), cfg, seed=0):
        assert not f.val.any() and not f.buffer.any()
