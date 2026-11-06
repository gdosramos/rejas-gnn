"""Non-neural baselines evaluated under exactly the same protocol as the battery.

Why this script exists
----------------------
A claim such as "the MLP matches GraphSAGE" is only informative if both beat the simplest
models a reviewer would try first. This script runs, with the *same* spatial folds
(``spatial_block_kfold``), seeds, training nodes, isotonic calibration on validation blocks and
pooled out-of-fold metrics as ``rejas_gnn.training.pipeline``:

* ``prevalence``      - constant base rate: the reference for the Brier skill score;
* ``table_road_cds``  - P(y | OSM road group, dead end) estimated by counting on training
                        nodes (12 cells, Laplace-smoothed): a two-variable lookup table;
* ``logreg_all``      - L2 logistic regression on the 28 geometry features;
* ``logreg_no_road``  - same without OSM road class (tests whether road tagging drives the signal);
* ``logreg_no_loc``   - same without ``dist_centroid_km`` (tests reliance on location).

Usage:  python scripts/run_baselines.py --seeds 42 43 44 45 46 47 48 49 50 51
Output: results/baselines/baselines_results.{csv,md} and one JSON per baseline.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from rejas_gnn.config import load_config
from rejas_gnn.data.graph_builder import build_urban_graph
from rejas_gnn.data.splits import spatial_block_kfold
from rejas_gnn.evaluation.metrics_and_calibration import fit_isotonic, full_report
from rejas_gnn.logging_utils import dump_json, environment_fingerprint, setup_logging
from rejas_gnn.metrics import best_f1_threshold

logger = logging.getLogger(__name__)
KEYS = ("pr_auc", "roc_auc", "f1", "mcc", "brier", "brier_skill", "ece")
Fitter = Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]]


def logreg(cols: list[int], X: np.ndarray, y: np.ndarray) -> Fitter:
    def fit(train_idx: np.ndarray, _unused: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        mu, sd = X[train_idx][:, cols].mean(0), X[train_idx][:, cols].std(0) + 1e-6
        m = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
        m.fit((X[train_idx][:, cols] - mu) / sd, y[train_idx])
        return lambda idx: m.predict_proba((X[idx][:, cols] - mu) / sd)[:, 1]
    return fit


def lookup(key: np.ndarray, y: np.ndarray, alpha: float = 2.0) -> Fitter:
    def fit(train_idx: np.ndarray, _unused: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        prior = y[train_idx].mean()
        df = pd.DataFrame({"k": key[train_idx], "y": y[train_idx]}).groupby("k")["y"].agg(["sum", "count"])
        rate = ((df["sum"] + alpha * prior) / (df["count"] + alpha)).to_dict()
        return lambda idx: np.array([rate.get(k, prior) for k in key[idx]])
    return fit


def prevalence(y: np.ndarray) -> Fitter:
    def fit(train_idx: np.ndarray, _unused: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        pi = y[train_idx].mean()
        return lambda idx: np.full(len(idx), pi)
    return fit


def run_seed(fitter: Fitter, xy: np.ndarray, y: np.ndarray, cfg, seed: int) -> dict:
    oof_raw, oof_cal = np.full(len(y), np.nan), np.full(len(y), np.nan)
    thr_raw, thr_cal = [], []
    for fm in spatial_block_kfold(xy, cfg.split, seed=seed, labels=y):
        tr, va, te = (np.flatnonzero(m.numpy()) for m in (fm.train, fm.val, fm.test))
        predict = fitter(tr, va)
        p_va, p_te = predict(va), predict(te)
        oof_raw[te] = p_te
        thr_raw.append(best_f1_threshold(y[va], p_va))
        if 0 < y[va].sum() < len(va) and np.ptp(p_va) > 0:
            iso = fit_isotonic(y[va], p_va)
            oof_cal[te] = np.clip(iso.predict(p_te), 0, 1)
            thr_cal.append(best_f1_threshold(y[va], np.clip(iso.predict(p_va), 0, 1)))
        else:  # constant predictor: nothing to calibrate
            oof_cal[te] = p_te
            thr_cal.append(thr_raw[-1])
    assert np.isfinite(oof_raw).all() and np.isfinite(oof_cal).all()
    return {"seed": seed,
            "raw": full_report(y, oof_raw, float(np.median(thr_raw))),
            "calibrated": full_report(y, oof_cal, float(np.median(thr_cal)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--out", type=Path, default=Path("results/baselines"))
    a = ap.parse_args()
    setup_logging(Path("results/logs"))
    cfg = load_config(a.config)
    cfg = replace(cfg, data=replace(cfg.data, feature_set="geometry"))
    ug = build_urban_graph(cfg.data, cfg.graph)
    X, y = ug.data.x.numpy(), ug.data.y.numpy()
    xy, names = ug.data.pos.numpy().astype(np.float64), ug.feature_names

    road = [i for i, n in enumerate(names) if n.startswith("via_") or n == "road_hierarchy"]
    allc = list(range(len(names)))
    seg = ug.segments
    grp = seg["tipo_via_osm"].where(
        seg["tipo_via_osm"].isin(["living_street", "service", "pedestrian", "footway", "residential"]), "arterial")
    key = (grp.astype(str) + "|" + X[:, names.index("es_culdesac")].astype(int).astype(str)).to_numpy()

    baselines: dict[str, Fitter] = {
        "prevalence": prevalence(y),
        "table_road_cds": lookup(key, y),
        "logreg_all": logreg(allc, X, y),
        "logreg_no_road": logreg([i for i in allc if i not in road], X, y),
        "logreg_no_loc": logreg([i for i in allc if names[i] != "dist_centroid_km"], X, y),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, fitter in baselines.items():
        per_seed = [run_seed(fitter, xy, y, cfg, s) for s in a.seeds]
        summ = {"baseline": name, "seeds": list(a.seeds)}
        for kind in ("raw", "calibrated"):
            for k in KEYS:
                v = np.array([r[kind][k] for r in per_seed], dtype=float)
                summ[f"{kind}_{k}_mean"], summ[f"{kind}_{k}_std"] = float(np.nanmean(v)), float(np.nanstd(v, ddof=1))
        rows.append(summ)
        dump_json({"environment": environment_fingerprint(), "summary": summ, "per_seed": per_seed},
                  a.out / f"{name}.json")
        logger.info("== %s | PR-AUC %.3f±%.3f | ROC-AUC %.3f±%.3f | BSS cal %.3f", name,
                    summ["raw_pr_auc_mean"], summ["raw_pr_auc_std"], summ["raw_roc_auc_mean"],
                    summ["raw_roc_auc_std"], summ["calibrated_brier_skill_mean"])
    t = pd.DataFrame(rows)
    t.to_csv(a.out / "baselines_results.csv", index=False)
    show = pd.DataFrame({"Baseline": t.baseline})
    for k, lab in (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")):
        show[lab] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(t[f"raw_{k}_mean"], t[f"raw_{k}_std"], strict=True)]
    show["Brier cal"] = t.calibrated_brier_mean.round(4)
    show["BSS cal"] = t.calibrated_brier_skill_mean.round(3)
    (a.out / "baselines_results.md").write_text(
        "# Non-neural baselines (same spatial folds, seeds and pooled OOF metrics as the battery)\n\n"
        + show.to_markdown(index=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
