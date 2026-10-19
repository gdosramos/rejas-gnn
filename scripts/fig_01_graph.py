"""Step 2 figures: the urban graph of PAC — what the model actually sees.

Produces results/figures/fig01_*.png and prints a short numeric summary.
Run:  python scripts/fig_01_graph.py [--feature-set geometry|geometry_socio]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rejas_gnn.config import DataConfig, GraphConfig  # noqa: E402
from rejas_gnn.data.graph_builder import build_urban_graph  # noqa: E402
from rejas_gnn.metrics import morans_i  # noqa: E402

OUT = Path("results/figures")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="geometry", choices=["geometry", "geometry_socio"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ug = build_urban_graph(DataConfig(feature_set=a.feature_set), GraphConfig(type="line"))
    seg, d = ug.segments, ug.data
    y = d.y.numpy()
    deg = torch.bincount(d.edge_index[0], minlength=d.num_nodes).numpy()

    # --- Figure A: the network and the closures
    fig, ax = plt.subplots(figsize=(9, 9))
    seg[seg.y == 0].plot(ax=ax, color="#c8c8c8", linewidth=0.6)
    seg[seg.is_condominio == 1].plot(ax=ax, color="#1f77b4", linewidth=1.4, label="condominio (no es objetivo)")
    seg[seg.y == 1].plot(ax=ax, color="#d62728", linewidth=1.6, label=f"cierre social (y=1, n={int(y.sum())})")
    ax.set_title("Pedro Aguirre Cerda: red vial OSM y cierres levantados en terreno")
    ax.set_axis_off(); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig01a_red_y_cierres.png", dpi=150); plt.close(fig)

    # --- Figure B: degree distribution of the line graph, by label
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, deg.max() + 2) - 0.5
    ax.hist(deg[y == 0], bins=bins, density=True, alpha=0.6, label="abierta")
    ax.hist(deg[y == 1], bins=bins, density=True, alpha=0.6, label="cierre")
    ax.set_xlabel("vecinos en el grafo dual (tramos que comparten esquina)"); ax.set_ylabel("densidad")
    ax.set_title("¿Los tramos cerrados tienen una posición distinta en la red?"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig01b_grado_dual.png", dpi=150); plt.close(fig)

    # --- Figure C: closure rate by road class and by cul-de-sac
    names = ug.feature_names
    X = d.x.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cls = [n for n in names if n.startswith("via_")]
    rates = [(n[4:], y[X[:, names.index(n)] == 1].mean(), int((X[:, names.index(n)] == 1).sum())) for n in cls]
    rates.sort(key=lambda t: -t[1])
    axes[0].barh([f"{n} (n={k})" for n, _, k in rates], [r for _, r, _ in rates], color="#d62728")
    axes[0].axvline(y.mean(), ls="--", color="k", label=f"tasa global {y.mean():.3f}")
    axes[0].set_xlabel("tasa de cierre"); axes[0].set_title("Por clase de vía OSM"); axes[0].legend()
    cul = X[:, names.index("es_culdesac")] == 1
    axes[1].bar(["con salida", "sin salida (cul-de-sac)"], [y[~cul].mean(), y[cul].mean()], color=["#999", "#d62728"])
    axes[1].set_ylabel("tasa de cierre"); axes[1].set_title(f"Calles sin salida (n={int(cul.sum())})")
    fig.tight_layout(); fig.savefig(OUT / "fig01c_tasa_por_tipo.png", dpi=150); plt.close(fig)

    mi = morans_i(y.astype(float), d.edge_index, d.num_nodes)
    print("\n=== Resumen del grafo urbano ===")
    print(f"tramos (nodos del grafo dual): {d.num_nodes} | cierres: {int(y.sum())} ({y.mean()*100:.1f}%)")
    print(f"intersecciones OSM: {ug.primal.number_of_nodes()} | aristas dual: {d.edge_index.size(1)//2}")
    print(f"grado medio dual: {deg.mean():.2f} | tramos aislados: {int((deg==0).sum())}")
    print(f"features ({a.feature_set}): {d.x.size(1)}")
    print(f"I de Moran de la etiqueta sobre el grafo dual: {mi:.3f}  (0 = sin autocorrelación)")
    print(f"tasa de cierre en cul-de-sac {y[cul].mean():.3f} vs resto {y[~cul].mean():.3f}")
    print(f"figuras en {OUT}/")


if __name__ == "__main__":
    main()
