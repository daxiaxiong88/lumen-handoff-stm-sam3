"""Dense-prediction metrics for depth and surface-normal benchmarking.

The benchmark subsystem was semantic-segmentation-only, so the Vision Banana
depth/normal examples each reimplemented ``absrel`` / ``angular_error`` inline.
These are the shared, tested versions — the numeric conventions match those
examples (``absrel`` masks ``gt > eps``; ``angular_error`` is the mean arccos of
the clamped dot product in degrees) and add the standard threshold metrics.

All functions operate on NumPy arrays, matching
:class:`lumen.models.zoo.PredictionResult` dense fields (depth ``HxW``, normals
``HxWx3``).
"""

from __future__ import annotations

import numpy as np


def _depth_valid(target: np.ndarray, mask: np.ndarray | None, eps: float) -> np.ndarray:
    valid = target > eps
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)
    return valid


def depth_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    eps: float = 1e-3,
) -> dict[str, float]:
    """Standard monocular-depth metrics over valid (``target > eps``) pixels.

    Returns ``abs_rel``, ``rmse``, ``delta1``/``delta2``/``delta3``
    (fraction with ``max(p/t, t/p) < 1.25**k``), and ``num_valid``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = _depth_valid(target, mask, eps)
    if not valid.any():
        return {
            "abs_rel": 0.0,
            "rmse": 0.0,
            "delta1": 0.0,
            "delta2": 0.0,
            "delta3": 0.0,
            "num_valid": 0.0,
        }
    p = pred[valid]
    t = target[valid]
    p_safe = np.clip(p, eps, None)
    ratio = np.maximum(p_safe / t, t / p_safe)
    return {
        "abs_rel": float(np.mean(np.abs(p - t) / t)),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
        "num_valid": float(valid.sum()),
    }


def abs_rel(pred: np.ndarray, target: np.ndarray, *, eps: float = 1e-3) -> float:
    """Mean absolute relative depth error over ``target > eps`` (example parity)."""
    return depth_metrics(pred, target, eps=eps)["abs_rel"]


def _unit_normals(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    unit: np.ndarray = v / np.clip(norm, eps, None)
    return unit


def angular_error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-pixel angular error (degrees) between two ``HxWx3`` normal maps."""
    p = _unit_normals(pred)
    t = _unit_normals(target)
    dot = np.clip(np.sum(p * t, axis=-1), -1.0, 1.0)
    degrees: np.ndarray = np.degrees(np.arccos(dot))
    return degrees


def normal_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Surface-normal metrics: mean/median angular error and threshold fractions."""
    err = angular_error_map(pred, target)
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
    else:
        valid = np.ones(err.shape, dtype=bool)
    if not valid.any():
        return {
            "mean_angular_error": 0.0,
            "median_angular_error": 0.0,
            "within_11_25": 0.0,
            "within_22_5": 0.0,
            "within_30": 0.0,
            "num_valid": 0.0,
        }
    e = err[valid]
    return {
        "mean_angular_error": float(np.mean(e)),
        "median_angular_error": float(np.median(e)),
        "within_11_25": float(np.mean(e < 11.25)),
        "within_22_5": float(np.mean(e < 22.5)),
        "within_30": float(np.mean(e < 30.0)),
        "num_valid": float(valid.sum()),
    }


def angular_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean angular error in degrees (example parity)."""
    return normal_metrics(pred, target)["mean_angular_error"]


__all__ = [
    "depth_metrics",
    "abs_rel",
    "angular_error_map",
    "normal_metrics",
    "angular_error",
]
