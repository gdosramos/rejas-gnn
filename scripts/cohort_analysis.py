"""Do the two registration cohorts of closures obey the same pattern?

Why this analysis exists
------------------------
The survey's ``año`` column is the year a closure was registered. Closures fall into two cohorts
that follow the two Chilean laws regulating street closures: 2012-2015 (109 closures, after Ley
20.499 of 2011, which requires a single access) and 2022-2025 (108 closures, after Ley 21.411 of
2022, which also allows streets with two accesses). If the model mostly learned the *legal
eligibility* of the first regime (dead ends), it should rank first-cohort closures well and
second-cohort closures poorly, and a model trained on one cohort should not transfer to the other.

Three outputs, all with the battery's spatial folds and seeds and the best model (logistic
regression on block A):

1. **Profile** of each cohort (dead ends, living streets, length).
2. **Stratified evaluation.** The model is trained on all closures; on the test blocks, the
   closures of one cohort are ranked against all open segments (the other cohort is left out).
3. **Cross-cohort transfer.** The model is trained with the closures of one cohort only (the
   other cohort's closures are removed from training, not relabelled as open) and evaluated on
   each cohort. Train c1 -> test c2 near train c2 -> test c2 means the pattern transfers.

ROC-AUC is the primary metric here: it does not depend on the cohort's prevalence, so it is
comparable across cohorts. PR-AUC is reported with its chance level.

Usage:  python scripts/cohort_analysis.py
Output: results/cohorts/cohort_analysis.md (+ .csv)
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from rejas_gnn.config import load_config
from rejas_gnn.data.graph_builder import build_urban_graph
from rejas_gnn.data.splits import spatial_block_kfold
from rejas_gnn.logging_utils import setup_logging

logger = logging.getLogger(__name__)
COHORTS = {"2012-2015": (2012, 2015), "2022-2025": (2022, 2025)}


def fit_predict(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    m = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
    m.fit((X[train] - mu) / sd, y[train])
    return m.predict_proba((X[test] - mu) / sd)[:, 1]


def scores(y_pos: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    """ROC-AUC, PR-AUC and PR-AUC chance level for a positives-vs-open subset."""
    return roc_auc_score(y_pos, p), average_precision_score(y_pos, p), float(y_pos.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--out", type=Path, default=Path("results/cohorts"))
    a = ap.parse_args()
    setup_logging(Path("results/logs"))
    logging.getLogger("rejas_gnn.data.splits").setLevel(logging.WARNING)
    cfg = load_config(a.config)
    cfg = replace(cfg, data=replace(cfg.data, feature_set="geometry"))
    ug = build_urban_graph(cfg.data, cfg.graph)
    X, y = ug.data.x.numpy(), ug.data.y.numpy()
    xy, names = ug.data.pos.numpy().astype(np.float64), ug.feature_names
    year = ug.segments["año"].to_numpy()

    cohort = np.full(len(y), "open", dtype=object)
    for c, (lo, hi) in COHORTS.items():
        cohort[(y == 1) & (year >= lo) & (year <= hi)] = c
    assert ((cohort != "open") == (y == 1)).all(), "every closure must belong to a cohort"

    # 1. Profile
    seg = ug.segments.assign(cohort=cohort, dead_end=X[:, names.index("es_culdesac")])
    profile = seg.groupby("cohort").agg(
        n=("y", "size"), dead_end_share=("dead_end", "mean"),
        living_street_share=("tipo_via_osm", lambda s: (s == "living_street").mean()),
        median_length_m=("distancia_m", "median")).reindex(["2012-2015", "2022-2025", "open"])

    # 2 and 3. Stratified evaluation and cross-cohort transfer
    rows = []
    for s in a.seeds:
        folds = spatial_block_kfold(xy, cfg.split, seed=s, labels=y)
        oof = {k: np.full(len(y), np.nan) for k in ("all", *COHORTS)}
        for fm in folds:
            tr = np.flatnonzero(fm.train.numpy())
            te = np.flatnonzero(fm.test.numpy())
            oof["all"][te] = fit_predict(X, y, tr, te)
            for c in COHORTS:  # train with the closures of cohort c only
                tr_c = tr[(cohort[tr] == "open") | (cohort[tr] == c)]
                oof[c][te] = fit_predict(X, y, tr_c, te)
        for train_on, p in oof.items():
            for test_on in COHORTS:
                keep = (cohort == "open") | (cohort == test_on)
                roc, pr, chance = scores(y[keep], p[keep])
                rows.append({"seed": s, "train_on": train_on, "test_on": test_on,
                             "roc_auc": roc, "pr_auc": pr, "pr_chance": chance})
    t = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out / "cohort_analysis.csv", index=False)

    agg = t.groupby(["train_on", "test_on"]).agg(
        roc_mean=("roc_auc", "mean"), roc_sd=("roc_auc", "std"),
        pr_mean=("pr_auc", "mean"), pr_sd=("pr_auc", "std"), chance=("pr_chance", "mean")).reset_index()
    order = {"all": 0, "2012-2015": 1, "2022-2025": 2}
    agg = agg.sort_values(["train_on", "test_on"], key=lambda c: c.map(order) if c.name == "train_on" else c)
    show = pd.DataFrame({
        "Trained on closures of": agg.train_on.str.replace("all", "both cohorts"),
        "Evaluated on closures of": agg.test_on,
        "ROC-AUC": [f"{m:.3f} ± {sd:.3f}" for m, sd in zip(agg.roc_mean, agg.roc_sd, strict=True)],
        "PR-AUC (chance)": [f"{m:.3f} ± {sd:.3f} ({c:.3f})"
                            for m, sd, c in zip(agg.pr_mean, agg.pr_sd, agg.chance, strict=True)],
    })
    # Paired transfer gap per seed: (train on target) - (train on the other cohort), on ROC-AUC
    gaps = []
    for target, other in (("2012-2015", "2022-2025"), ("2022-2025", "2012-2015")):
        own = t[(t.train_on == target) & (t.test_on == target)].set_index("seed").roc_auc
        cross = t[(t.train_on == other) & (t.test_on == target)].set_index("seed").roc_auc
        d = (own - cross).to_numpy()
        boot = np.random.default_rng(0).choice(d, size=(10_000, len(d))).mean(1)
        gaps.append({"Evaluated on": target, "Own cohort − other cohort (ROC-AUC)": f"{d.mean():+.3f}",
                     "95 % CI": f"[{np.quantile(boot, .025):+.3f}, {np.quantile(boot, .975):+.3f}]"})
    md = ["# Closures by registration cohort (logistic regression, block A, spatial folds)\n",
          "## Profile\n", profile.round(3).to_markdown(), "\n",
          "## Stratified evaluation and cross-cohort transfer (mean ± SD over seeds)\n",
          show.to_markdown(index=False), "\n",
          "## Transfer gap: training on the target cohort vs on the other cohort\n",
          pd.DataFrame(gaps).to_markdown(index=False), "\n"]
    (a.out / "cohort_analysis.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
