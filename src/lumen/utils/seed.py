"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs.

    Args:
        seed: The seed to apply across all RNGs.
        deterministic: If True, also request deterministic cuDNN algorithms
            (slower, but bit-reproducible on CUDA).

    Returns:
        The seed, for convenient logging.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


__all__ = ["seed_everything"]
