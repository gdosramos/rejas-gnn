"""Command-line interface: ``rejas-gnn run --config configs/sage.yaml [--set train.epochs=5]``."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from rejas_gnn.config import load_config
from rejas_gnn.logging_utils import setup_logging
from rejas_gnn.pipeline import run


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise argparse.ArgumentTypeError(f"override must be section.key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)  # parse numbers / lists / booleans
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rejas-gnn")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run one experiment")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--set", nargs="*", default=[], help="overrides section.key=value")
    p_run.add_argument("--log-dir", type=Path, default=Path("results/logs"))
    args = parser.parse_args(argv)

    setup_logging(args.log_dir)
    cfg = load_config(args.config, _parse_overrides(args.set))
    logging.getLogger(__name__).info("Config: %s", cfg)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
