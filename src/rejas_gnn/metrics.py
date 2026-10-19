"""Evaluation metrics for imbalanced binary node classification plus spatial diagnostics."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    """Threshold on ``p`` maximising F1 (chosen on validation only)."""
    assert y_true.shape == p.shape
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    f1s = [f1_score(y_true, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def classification_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """ROC-AUC, PR-AUC (primary), F1/precision/recall at ``threshold``, MCC, Brier."""
    assert y_true.shape == p.shape and p.min() >= 0 and p.max() <= 1
    pred = (p >= threshold).astype(int)
    out: dict[str, float] = {
        "n": float(len(y_true)),
        "positives": float(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, p)) if y_true.sum() else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, p)) if 0 < y_true.sum() < len(y_true) else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "brier": float(brier_score_loss(y_true, p)),
        "threshold": float(threshold),
    }
    return out


def morans_i(values: np.ndarray, edge_index: torch.Tensor, num_nodes: int) -> float:
    """Global Moran's I of ``values`` on an unweighted undirected graph (binary W).

    ``I = (N / S0) * (sum_ij w_ij z_i z_j) / (sum_i z_i^2)`` with ``z = values - mean``.
    Used to (a) quantify spatial autocorrelation of the label (justifying spatial CV) and
    (b) check that model residuals are not spatially structured.
    """
    assert values.shape == (num_nodes,)
    z = values.astype(np.float64) - values.mean()
    src, dst = edge_index.numpy()
    s0 = float(len(src))
    num = float((z[src] * z[dst]).sum())
    den = float((z**2).sum())
    if den == 0 or s0 == 0:
        return float("nan")
    return (num_nodes / s0) * num / den
