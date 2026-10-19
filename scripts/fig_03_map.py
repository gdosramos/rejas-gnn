"""Step 6 figures: the calibrated probability map, its uncertainty, calibration, and candidates.

Reads results/battery/<experiment>_p_closure.gpkg and <experiment>_reliability.csv.
Run:  python scripts/fig_03_map.py [--experiment geometry__mlp] [--top 25]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

OUT = Path("results/figures")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="geometry__mlp")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = Path("results/battery")
    g = gpd.read_file(base / f"{a.experiment}_p_closure.gpkg")
    rel = pd.read_csv(base / f"{a.experiment}_reliability.csv")
    assert {"p_closure", "p_closure_raw", "p_closure_std_seeds", "y"} <= set(g.columns)
    y, p = g["y"].to_numpy(), g["p_closure"].to_numpy()

    # --- A: probability map + uncertainty
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    g.plot(ax=axes[0], column="p_closure", cmap="magma_r", linewidth=1.0, legend=True,
           legend_kwds={"shrink": 0.6, "label": "P(cierre) calibrada, fuera de muestra"})
    g[g.y == 1].plot(ax=axes[0], color="none", edgecolor="#00b4ff", linewidth=0.5)
    axes[0].set_title(f"{a.experiment}: probabilidad de cierre (contorno celeste = cierre real)")
    g.plot(ax=axes[1], column="p_closure_std_seeds", cmap="viridis", linewidth=1.0, legend=True,
           legend_kwds={"shrink": 0.6, "label": "DE entre 10 semillas"})
    axes[1].set_title("Incertidumbre: dónde las 10 semillas discrepan")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout(); fig.savefig(OUT / f"fig03a_mapa_{a.experiment}.png", dpi=150); plt.close(fig)

    # --- B: reliability curve + distribution of p by class
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="calibración perfecta")
    axes[0].plot(rel.p_mean, rel.y_rate, "o-", color="#d62728", label="modelo calibrado (10 bins)")
    axes[0].set_xlabel("probabilidad predicha (media del bin)"); axes[0].set_ylabel("frecuencia observada de cierre")
    axes[0].set_xlim(0, max(0.6, rel.p_mean.max() * 1.1)); axes[0].set_ylim(0, max(0.6, rel.y_rate.max() * 1.1))
    axes[0].set_title("Curva de confiabilidad"); axes[0].legend()
    bins = np.linspace(0, 1, 41)
    axes[1].hist(p[y == 0], bins=bins, alpha=0.6, density=True, label="abierta")
    axes[1].hist(p[y == 1], bins=bins, alpha=0.6, density=True, label="cierre")
    axes[1].set_xlabel("P(cierre) calibrada"); axes[1].set_ylabel("densidad"); axes[1].legend()
    axes[1].set_title(f"Separación: ROC-AUC {roc_auc_score(y, p):.3f} | PR-AUC {average_precision_score(y, p):.3f}")
    fig.tight_layout(); fig.savefig(OUT / f"fig03b_calibracion_{a.experiment}.png", dpi=150); plt.close(fig)

    # --- C: candidates = open segments with the highest calibrated probability
    cand = g[g.y == 0].sort_values("p_closure", ascending=False).head(a.top)
    cols = ["arista_id", "nombre_calle", "tipo_via_osm", "estado_cierre", "p_closure", "p_closure_std_seeds"]
    cols = [c for c in cols if c in cand.columns]
    cand[cols].to_csv(OUT / f"candidatos_{a.experiment}.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 9))
    g.plot(ax=ax, color="#d9d9d9", linewidth=0.6)
    g[g.y == 1].plot(ax=ax, color="#d62728", linewidth=1.2, label="cierre real")
    cand.plot(ax=ax, color="#ff9f00", linewidth=2.5, label=f"top-{a.top} abiertos por P(cierre)")
    ax.set_title("Candidatos: tramos abiertos que el modelo cree que 'deberían' estar cerrados")
    ax.set_axis_off(); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / f"fig03c_candidatos_{a.experiment}.png", dpi=150); plt.close(fig)

    q = np.quantile(p, [0.5, 0.9, 0.99])
    print(f"\n=== {a.experiment} ===")
    print(f"P(cierre) mediana {q[0]:.3f} | p90 {q[1]:.3f} | p99 {q[2]:.3f} | prevalencia real {y.mean():.3f}")
    print(f"ROC-AUC {roc_auc_score(y, p):.3f} | PR-AUC {average_precision_score(y, p):.3f} (promedio de 10 semillas, OOF)")
    print(f"cierres reales con P >= 0.25: {int(((p >= 0.25) & (y == 1)).sum())} de {int(y.sum())}")
    print(f"tramos abiertos con P >= 0.25: {int(((p >= 0.25) & (y == 0)).sum())}  <- candidatos")
    print("\ntop-10 candidatos (abiertos):")
    print(cand[cols].head(10).to_string(index=False))
    print(f"\nfiguras en {OUT}/ | lista completa en candidatos_{a.experiment}.csv")


if __name__ == "__main__":
    main()
