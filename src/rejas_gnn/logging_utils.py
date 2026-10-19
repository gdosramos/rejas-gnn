"""Audit logging: every run writes a timestamped, machine-parseable log to ``results/``."""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(log_dir: Path, level: int = logging.INFO) -> Path:
    """Configure root logging to console and to ``log_dir/run_<UTC timestamp>.log``.

    Returns
    -------
    Path
        The log file path.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = log_dir / f"run_{stamp}.log"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)
    logging.getLogger(__name__).info("Log file: %s", log_file)
    return log_file


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - not a git checkout or git missing
        return None


def environment_fingerprint() -> dict[str, Any]:
    """Collect versions and hardware info for the audit trail."""
    import torch_geometric

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git_commit(),
    }


def dump_json(obj: Any, path: Path) -> None:
    """Write ``obj`` as pretty JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
