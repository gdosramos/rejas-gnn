"""Regression tests for calibration metrics (bugs found in the v0.2 audit)."""
import numpy as np

from rejas_gnn.evaluation.metrics_and_calibration import brier_skill_score, fit_isotonic
from rejas_gnn.metrics import best_f1_threshold


def test_bss_constant_predictor_is_zero():
    y = np.array([0] * 92 + [1] * 8)
    assert abs(brier_skill_score(y, np.full(100, y.mean()))) < 1e-12


def test_bss_perfect_and_worse_than_constant():
    y = np.array([0] * 90 + [1] * 10)
    assert brier_skill_score(y, y.astype(float)) == 1.0
    assert brier_skill_score(y, np.full(100, 0.5)) < 0  # uninformative but miscalibrated


def test_threshold_is_chosen_on_calibrated_scale():
    """A raw-scale threshold applied to isotonic outputs can predict no positives at all."""
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.08).astype(int)
    raw = np.clip(0.6 + 0.3 * y + 0.1 * rng.standard_normal(400), 0, 1)  # class-weighted, inflated
    iso = fit_isotonic(y, raw)
    cal = iso.predict(raw)
    thr_raw, thr_cal = best_f1_threshold(y, raw), best_f1_threshold(y, cal)
    assert (cal >= thr_cal).any()
    assert thr_cal < thr_raw
