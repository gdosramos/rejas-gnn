"""Step 3 figures: spatial block cross-validation — why we cannot split at random.

Produces results/figures/fig02_*.png and prints the composition of each fold.
Run:  python scripts/fig_02_splits.py [--seed 42]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from rejas_gnn.config import DataConfig, GraphConfig, SplitConfig  # noqa: E402
from rejas_gnn.data.graph_builder import build_urban_graph  # noqa: E402
from rejas_gnn.data.splits import spatial_block_kfold, spatial_blocks  # noqa: E402

OUT = Path("results/figures")
COLORS = {"train": "#9e9e9e", "val": "#1f77b4", "test": "#d62728", "buffer": "#ffbf00"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ug = build_urban_graph(DataConfig(feature_set="geometry", covariates_csv=None), GraphConfig(type="line"))
    seg = ug.segments
    xy = ug.data.pos.numpy().astype(np.float64)
    y = ug.data.y.numpy()
    cfg = SplitConfig()

    # --- A: the 20 spatial blocks
    blocks = spatial_blocks(xy, cfg.n_blocks, a.seed)
    fig, ax = plt.subplots(figsize=(8, 8))
    seg.plot(ax=ax, column=blocks, categorical=True, cmap="tab20", linewidth=0.8, legend=False)
    for b in range(cfg.n_blocks):
        cx, cy = xy[blocks == b].mean(0)
        ax.text(cx, cy, str(b), ha="center", va="center", fontsize=9, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    ax.set_title(f"{cfg.n_blocks} bloques espaciales (k-means sobre coordenadas, semilla {a.seed})")
    ax.set_axis_off(); fig.tight_layout(); fig.savefig(OUT / "fig02a_bloques.png", dpi=150); plt.close(fig)

    # --- B: the 5 folds, one panel each
    folds = spatial_block_kfold(xy, cfg, seed=a.seed, labels=y)
    fig, axes = plt.subplots(1, cfg.n_folds, figsize=(4 * cfg.n_folds, 4.6))
    print(f"\n=== Composición de los folds (semilla {a.seed}, buffer {cfg.buffer_m:.0f} m) ===")
    print(f"{'fold':>4} {'train':>6} {'val':>5} {'test':>5} {'buffer':>7} {'cierres test':>13} {'cierres train':>14}")
    for ax, fm in zip(axes, folds, strict=True):
        role = np.full(len(seg), "train", dtype=object)
        for k in ("val", "buffer", "test"):
            role[getattr(fm, k).numpy()] = k
        for k, lw in (("train", 0.5), ("val", 0.9), ("buffer", 1.1), ("test", 1.1)):
            seg[role == k].plot(ax=ax, color=COLORS[k], linewidth=lw)
        seg[(y == 1) & fm.test.numpy()].plot(ax=ax, color="black", linewidth=1.6)
        tr, va, te, bu = (int(getattr(fm, k).sum()) for k in ("train", "val", "test", "buffer"))
        ax.set_title(f"fold {fm.fold}: test {te} tramos / {int(y[fm.test.numpy()].sum())} cierres", fontsize=10)
        ax.set_axis_off()
        print(f"{fm.fold:>4} {tr:>6} {va:>5} {te:>5} {bu:>7} {int(y[fm.test.numpy()].sum()):>13} {int(y[fm.train.numpy()].sum()):>14}")
    handles = [Line2D([0], [0], color=c, lw=3) for c in COLORS.values()] + [Line2D([0], [0], color="black", lw=3)]
    fig.legend(handles, [*COLORS.keys(), "cierre en test"], loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Validación espacial por bloques con buffer: el modelo nunca ve etiquetas a < 200 m de un tramo de test")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95)); fig.savefig(OUT / "fig02b_folds.png", dpi=150); plt.close(fig)

    # --- C: random split vs spatial split — distance from each test segment to nearest labelled train segment
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(a.seed)
    rand_test = rng.random(len(xy)) < 0.2
    d_rand, _ = cKDTree(xy[~rand_test]).query(xy[rand_test])
    d_spat = np.concatenate([cKDTree(xy[fm.train.numpy() | fm.val.numpy()]).query(xy[fm.test.numpy()])[0] for fm in folds])
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 600, 41)
    ax.hist(d_rand, bins=bins, alpha=0.6, label=f"split aleatorio (mediana {np.median(d_rand):.0f} m)")
    ax.hist(d_spat, bins=bins, alpha=0.6, label=f"bloques + buffer (mediana {np.median(d_spat):.0f} m)")
    ax.axvline(cfg.buffer_m, ls="--", color="k"); ax.set_xlabel("distancia del tramo de test al tramo etiquetado más cercano (m)")
    ax.set_ylabel("tramos de test"); ax.set_title("Por qué el split aleatorio es trampa"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig02c_distancia_fuga.png", dpi=150); plt.close(fig)
    print(f"\nsplit aleatorio: {np.mean(d_rand < 100)*100:.0f}% de los tramos de test tienen un tramo etiquetado a < 100 m")
    print(f"bloques+buffer:  {np.mean(d_spat < 100)*100:.0f}% (mínimo {d_spat.min():.0f} m)")
    print(f"figuras en {OUT}/")


if __name__ == "__main__":
    main()
