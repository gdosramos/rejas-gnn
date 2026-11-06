"""Paired comparisons between models evaluated on identical spatial partitions.

Why paired
----------
Every model in the battery and in the baselines is evaluated with the same seeds, and the seed
fixes the spatial block partition. Seed-to-seed variation is therefore mostly variation of the
*partition* (which neighbourhoods are held out), shared by all models. Comparing unpaired
means +- SD ignores that shared component and can hide or invent differences. Here each claim
"A beats B" is tested on the per-seed differences d_s = metric_A(s) - metric_B(s):

* mean difference and a 95 % percentile bootstrap CI over seeds (10 000 resamples);
* exact two-sided Wilcoxon signed-rank p-value;
* how many seeds A wins.

With 10 seeds the test has limited power: a non-significant result means "not shown", not
"equal". The partition is one source of uncertainty; the finite set of 2 790 segments is another
and is not covered here.

Usage:  python scripts/paired_comparison.py --battery results/battery --baselines results/baselines
Output: results/paired/paired_comparisons.{csv,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# (label, run A, run B): runs are "<feature_set>__<model>" files of the battery or baseline names.
COMPARISONS: list[tuple[str, str, str]] = [
    ("Linear model vs MLP (block A)", "logreg_all", "geometry__mlp"),
    ("Linear model vs GraphSAGE (block A)", "logreg_all", "geometry__sage"),
    ("MLP vs GraphSAGE (block A)", "geometry__mlp", "geometry__sage"),
    ("MLP vs 2-variable table", "geometry__mlp", "table_road_cds"),
    ("Block A vs A+B, MLP", "geometry__mlp", "geometry_socio__mlp"),
    ("Block A vs A+B, GraphSAGE", "geometry__sage", "geometry_socio__sage"),
    ("GraphSAGE vs GCN (block A)", "geometry__sage", "geometry__gcn"),
    ("Linear: with vs without road class", "logreg_all", "logreg_no_road"),
    ("Linear: with vs without distance to centroid", "logreg_all", "logreg_no_loc"),
]
METRICS = (("raw", "pr_auc", "PR-AUC"), ("raw", "roc_auc", "ROC-AUC"), ("calibrated", "brier_skill", "BSS cal"))


def load_runs(dirs: list[Path]) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            obj = json.loads(f.read_text(encoding="utf-8"))
            if "per_seed" in obj:
                runs[f.stem] = {int(r["seed"]): r for r in obj["per_seed"]}
    return runs


def paired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = 10_000) -> dict[str, float]:
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    p = float(wilcoxon(d, alternative="two-sided", method="exact").pvalue) if np.any(d != 0) else 1.0
    return {"diff": float(d.mean()), "ci_lo": float(np.quantile(boot, 0.025)), "ci_hi": float(np.quantile(boot, 0.975)),
            "p_wilcoxon": p, "wins": int((d > 0).sum()), "n": len(d)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", type=Path, default=Path("results/battery"))
    ap.add_argument("--baselines", type=Path, default=Path("results/baselines"))
    ap.add_argument("--out", type=Path, default=Path("results/paired"))
    a = ap.parse_args()
    runs = load_runs([a.battery, a.baselines])
    rng = np.random.default_rng(0)
    rows = []
    for label, ra, rb in COMPARISONS:
        if ra not in runs or rb not in runs:
            print(f"skip {label}: missing {ra if ra not in runs else rb}")
            continue
        seeds = sorted(set(runs[ra]) & set(runs[rb]))
        for kind, key, name in METRICS:
            va = np.array([runs[ra][s][kind].get(key, np.nan) for s in seeds], dtype=float)
            vb = np.array([runs[rb][s][kind].get(key, np.nan) for s in seeds], dtype=float)
            if np.isnan(va).any() or np.isnan(vb).any():
                continue
            rows.append({"comparison": label, "A": ra, "B": rb, "metric": name, **paired(va, vb, rng)})
    t = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out / "paired_comparisons.csv", index=False)
    show = pd.DataFrame({
        "Comparison (A vs B)": t.comparison, "Metric": t.metric,
        "Mean diff A−B": t["diff"].map("{:+.3f}".format),
        "95 % CI": [f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(t.ci_lo, t.ci_hi, strict=True)],
        "Wilcoxon p": t.p_wilcoxon.map("{:.3f}".format),
        "A wins": [f"{w}/{n}" for w, n in zip(t.wins, t.n, strict=True)],
    })
    (a.out / "paired_comparisons.md").write_text(
        "# Paired comparisons over seeds (same spatial partitions)\n\n"
        "A difference is supported when its 95 % CI excludes 0. Ten seeds give limited power.\n\n"
        + show.to_markdown(index=False) + "\n", encoding="utf-8")
    print((a.out / "paired_comparisons.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
