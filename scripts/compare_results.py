"""Aggregate ``results/*/summary.json`` into a Markdown comparison table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/comparison.md"))
    args = ap.parse_args()
    rows = []
    for f in sorted(args.results.glob("*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        a = s["aggregate"]
        rows.append({
            "experiment": f.parent.name, "model": s["model"], "graph": s["graph"],
            "PR-AUC": f"{a['pr_auc']['mean']:.3f} ± {a['pr_auc']['std']:.3f}",
            "ROC-AUC": f"{a['roc_auc']['mean']:.3f} ± {a['roc_auc']['std']:.3f}",
            "F1": f"{a['f1']['mean']:.3f} ± {a['f1']['std']:.3f}",
            "MCC": f"{a['mcc']['mean']:.3f} ± {a['mcc']['std']:.3f}",
            "Brier": f"{a['brier']['mean']:.3f}",
            "pooled OOF PR-AUC": f"{s['oof_pooled']['pr_auc']:.3f}",
            "resid. Moran I": f"{s['residual_morans_i']:.3f}",
            "_sort": a["pr_auc"]["mean"],
        })
    if not rows:
        raise SystemExit("no summary.json found")
    df = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
    md = "# Model comparison (spatial block CV, 5 folds x 5 seeds, test metrics)\n\n" + df.to_markdown(index=False) + "\n"
    args.out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
