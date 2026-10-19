"""Build and cache the PyG dataset without training (``make data``)."""
from __future__ import annotations

import argparse
from pathlib import Path

from rejas_gnn.config import load_config
from rejas_gnn.logging_utils import setup_logging
from rejas_gnn.pipeline import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    setup_logging(Path("results/logs"))
    cfg = load_config(args.config)
    cache = Path("data/processed") / f"dataset_{cfg.graph.type}_{cfg.data.feature_set}_{Path(cfg.data.raw_edges_csv).stem}.pt"
    data, _ = build_dataset(cfg, cache=cache)
    print(f"nodes={data.num_nodes} edges={data.edge_index.size(1)} features={data.x.size(1)}")


if __name__ == "__main__":
    main()
