"""End-to-end training on a synthetic graph: runs, is deterministic, outputs sane metrics."""
import numpy as np
import torch

from rejas_gnn.metrics import best_f1_threshold, classification_metrics, morans_i
from rejas_gnn.train import run_experiment, standardise


def test_standardise_uses_train_only():
    x = torch.tensor([[1.0], [3.0], [100.0]])
    mask = torch.tensor([True, True, False])
    z = standardise(x, mask)
    assert torch.allclose(z[:2].mean(), torch.tensor(0.0), atol=1e-6)


def test_metrics_basic():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = classification_metrics(y, p, 0.5)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0 and m["f1"] == 1.0
    assert 0.0 <= best_f1_threshold(y, p) <= 1.0


def test_morans_i_positive_for_clustered_label(random_graph):
    i = morans_i(random_graph.y.numpy().astype(float), random_graph.edge_index, 400)
    assert i > 0.3


def test_run_experiment_is_deterministic(random_graph, small_cfg):
    r1 = run_experiment(random_graph, small_cfg)
    r2 = run_experiment(random_graph, small_cfg)
    assert np.allclose(r1["oof_prob_mean"], r2["oof_prob_mean"])
    assert r1["aggregate"]["pr_auc"]["mean"] == r2["aggregate"]["pr_auc"]["mean"]
    assert 0.0 <= r1["aggregate"]["roc_auc"]["mean"] <= 1.0
    assert len(r1["per_fold"]) == small_cfg.split.n_folds * len(small_cfg.train.seeds)
