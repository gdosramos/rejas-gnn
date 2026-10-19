"""Global determinism controls.

All stochastic components of the pipeline (NumPy, Python ``random``, PyTorch CPU/CUDA,
PyTorch Geometric) are seeded through :func:`set_global_seed`. Deterministic kernels are
requested from PyTorch; when a deterministic implementation does not exist (some CUDA
scatter kernels), PyTorch emits a warning instead of failing because ``warn_only=True``.
"""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch
from torch_geometric import seed_everything

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every random number generator used in the pipeline.

    Parameters
    ----------
    seed:
        Non-negative integer seed.
    deterministic:
        If ``True``, also request deterministic algorithms from PyTorch and cuDNN and set
        ``CUBLAS_WORKSPACE_CONFIG`` (required for deterministic cuBLAS on CUDA >= 10.2).

    Raises
    ------
    ValueError
        If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)  # PyG wrapper: random, numpy, torch
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d (deterministic=%s)", seed, deterministic)
