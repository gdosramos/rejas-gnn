"""Phase-2 experiment battery: ablation × architectures × seeds with spatial block CV.

For every (feature_set, model, seed) the pipeline runs ``n_folds`` spatial folds, fits an
isotonic calibrator on the validation blocks of each fold, applies it to that fold's test
blocks, and stores raw and calibrated out-of-fold probabilities for every segment. Results
are aggregated as mean ± SD over seeds and written to ``results/battery/``.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rejas_gnn.config import ExperimentConfig, FeatureSet, ModelName
from rejas_gnn.data.graph_builder import UrbanGraph, build_urban_graph
from rejas_gnn.data.splits import spatial_block_kfold
from rejas_gnn.evaluation.metrics_and_calibration import (
    export_probability_map,
    fit_isotonic,
    full_report,
    reliability_table,
)
from rejas_gnn.logging_utils import dump_json, environment_fingerprint
from rejas_gnn.metrics import morans_i
from rejas_gnn.train import train_one_fold

logger = logging.getLogger(__name__)
METRIC_KEYS = ("pr_auc", "roc_auc", "f1", "precision", "recall", "mcc", "brier", "ece")


def run_seed(ug: UrbanGraph, cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    """Spatial CV for one seed; returns test metrics (raw + calibrated) and OOF vectors."""
    data = ug.data
    y = data.y.numpy()
    xy = data.pos.numpy().astype(np.float64)
    oof_raw = np.full(data.num_nodes, np.nan)
    oof_cal = np.full(data.num_nodes, np.nan)
    thresholds: list[float] = []
    for fm in spatial_block_kfold(xy, cfg.split, seed=seed, labels=y):
        r = train_one_fold(data, fm, cfg, seed)
        oof_raw[r.test_index] = r.test_prob
        yv = y[r.val_index]
        if len(yv) and 0 < yv.sum() < len(yv):
            iso = fit_isotonic(yv, r.val_prob)
            oof_cal[r.test_index] = np.clip(iso.predict(r.test_prob), 0.0, 1.0)  # guard fp round-off
        else:  # degenerate validation set: no calibration possible
            oof_cal[r.test_index] = r.test_prob
        thresholds.append(r.threshold)
    assert np.isfinite(oof_raw).all() and np.isfinite(oof_cal).all(), "incomplete OOF coverage"
    thr = float(np.median(thresholds))
    return {
        "seed": seed,
        "raw": full_report(y, oof_raw, thr),
        "calibrated": full_report(y, oof_cal, thr),
        "residual_morans_i": morans_i(y - oof_cal, data.edge_index, data.num_nodes),
        "oof_raw": oof_raw,
        "oof_cal": oof_cal,
    }


def run_battery(cfg: ExperimentConfig, feature_sets: tuple[FeatureSet, ...],
                models: tuple[ModelName, ...], seeds: tuple[int, ...],
                out_dir: Path) -> pd.DataFrame:
    """Run the full ablation battery and export tables, JSON summaries and probability maps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for fs in feature_sets:
        cfg_fs = replace(cfg, data=replace(cfg.data, feature_set=fs))
        ug = build_urban_graph(cfg_fs.data, cfg_fs.graph)
        y = ug.data.y.numpy()
        for mn in models:
            cfg_run = replace(cfg_fs, model=replace(cfg_fs.model, name=mn), train=replace(cfg_fs.train, seeds=seeds))
            tag = f"{fs}__{mn}"
            done = out_dir / f"{tag}.json"
            if done.exists():  # resume: reuse a finished experiment with the same seeds
                prev = json.loads(done.read_text(encoding="utf-8"))
                if prev["summary"].get("seeds") == list(seeds):
                    logger.info("== %s already done; skipping (delete %s to rerun)", tag, done)
                    rows.append(prev["summary"])
                    continue
            t0 = time.time()
            per_seed = [run_seed(ug, cfg_run, s) for s in seeds]
            oof_cal = np.mean([r["oof_cal"] for r in per_seed], axis=0)
            oof_raw = np.mean([r["oof_raw"] for r in per_seed], axis=0)
            summary: dict[str, Any] = {"feature_set": fs, "model": mn, "n_features": int(ug.data.x.size(1)),
                                       "seeds": list(seeds), "seconds": round(time.time() - t0, 1)}
            for kind in ("raw", "calibrated"):
                for k in METRIC_KEYS:
                    vals = np.array([r[kind][k] for r in per_seed], dtype=float)
                    summary[f"{kind}_{k}_mean"] = float(np.nanmean(vals))
                    summary[f"{kind}_{k}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary["residual_morans_i_mean"] = float(np.mean([r["residual_morans_i"] for r in per_seed]))
            summary["ensemble_pr_auc"] = full_report(y, oof_cal, 0.5)["pr_auc"]
            summary["ensemble_brier"] = full_report(y, oof_cal, 0.5)["brier"]
            rows.append(summary)
            dump_json({"config": cfg_run.to_dict(), "environment": environment_fingerprint(),
                       "feature_names": ug.feature_names, "summary": summary,
                       "per_seed": [{k: v for k, v in r.items() if not k.startswith("oof")} for r in per_seed]},
                      out_dir / f"{tag}.json")
            reliability_table(y, oof_cal).to_csv(out_dir / f"{tag}_reliability.csv", index=False)
            export_probability_map(ug.segments, oof_cal, out_dir / f"{tag}_p_closure.gpkg",
                                   extra={"p_closure_raw": oof_raw, "p_closure_std_seeds":
                                          np.std([r["oof_cal"] for r in per_seed], axis=0)})
            export_probability_map(ug.segments, oof_cal, out_dir / f"{tag}_p_closure.geojson")
            logger.info("== %s | PR-AUC %.3f±%.3f | ROC-AUC %.3f±%.3f | Brier %.3f→%.3f (cal) | ECE %.3f→%.3f",
                        tag, summary["raw_pr_auc_mean"], summary["raw_pr_auc_std"], summary["raw_roc_auc_mean"],
                        summary["raw_roc_auc_std"], summary["raw_brier_mean"], summary["calibrated_brier_mean"],
                        summary["raw_ece_mean"], summary["calibrated_ece_mean"])
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "battery_results.csv", index=False)
    _write_markdown(table, out_dir / "battery_results.md")
    return table


def _write_markdown(t: pd.DataFrame, path: Path) -> None:
    cols = {"feature_set": "Features", "model": "Model", "n_features": "#feat"}
    show = t[list(cols)].rename(columns=cols).copy()
    for k, lab in (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("f1", "F1"), ("mcc", "MCC")):
        show[lab] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(t[f"raw_{k}_mean"], t[f"raw_{k}_std"], strict=True)]
    show["Brier raw→cal"] = [f"{a:.3f} → {b:.3f}" for a, b in zip(t.raw_brier_mean, t.calibrated_brier_mean, strict=True)]
    show["ECE raw→cal"] = [f"{a:.3f} → {b:.3f}" for a, b in zip(t.raw_ece_mean, t.calibrated_ece_mean, strict=True)]
    show["Moran I resid."] = t.residual_morans_i_mean.round(3)
    show = show.sort_values("PR-AUC", ascending=False)
    path.write_text("# Ablation battery (spatial block CV, mean ± SD over seeds)\n\n" + show.to_markdown(index=False) + "\n",
                    encoding="utf-8")
