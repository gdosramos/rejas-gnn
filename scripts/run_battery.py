"""Run the full Phase-2 battery: ablation A/B × architectures × seeds 42–51 (``make run-all``)."""
from __future__ import annotations

import argparse
from pathlib import Path

from rejas_gnn.config import load_config
from rejas_gnn.logging_utils import setup_logging
from rejas_gnn.training.pipeline import run_battery


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--models", nargs="+", default=["mlp", "gcn", "sage", "gat"])
    ap.add_argument("--feature-sets", nargs="+", default=["geometry", "geometry_socio"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--out", type=Path, default=Path("results/battery"))
    a = ap.parse_args()
    setup_logging(Path("results/logs"))
    cfg = load_config(a.config)
    run_battery(cfg, tuple(a.feature_sets), tuple(a.models), tuple(a.seeds), a.out)


if __name__ == "__main__":
    main()
