"""Tests for depth and surface-normal benchmark metrics."""

from __future__ import annotations

import numpy as np

from lumen.benchmark.dense_metrics import (
    abs_rel,
    angular_error,
    angular_error_map,
    depth_metrics,
    normal_metrics,
)


class TestDepthMetrics:
    def test_perfect_prediction(self) -> None:
        target = np.array([[1.0, 2.0], [3.0, 4.0]])
        m = depth_metrics(target.copy(), target)
        assert m["abs_rel"] == 0.0
        assert m["rmse"] == 0.0
        assert m["delta1"] == 1.0
        assert m["num_valid"] == 4.0

    def test_double_prediction(self) -> None:
        target = np.array([[1.0, 2.0], [4.0, 8.0]])
        m = depth_metrics(2.0 * target, target)
        # |2t - t| / t == 1 everywhere
        assert m["abs_rel"] == 1.0
        # ratio max(2, 0.5) = 2.0 > 1.25**3 -> all thresholds 0
        assert m["delta1"] == 0.0
        assert m["delta3"] == 0.0

    def test_zero_target_pixels_are_masked_out(self) -> None:
        target = np.array([[0.0, 2.0]])  # first pixel invalid
        pred = np.array([[999.0, 2.0]])  # bad value on the invalid pixel
        m = depth_metrics(pred, target)
        assert m["num_valid"] == 1.0
        assert m["abs_rel"] == 0.0  # only the correct valid pixel counts

    def test_explicit_mask(self) -> None:
        target = np.array([[1.0, 1.0]])
        pred = np.array([[1.0, 5.0]])
        mask = np.array([[True, False]])
        assert depth_metrics(pred, target, mask=mask)["abs_rel"] == 0.0

    def test_abs_rel_convenience_matches(self) -> None:
        target = np.array([[1.0, 2.0]])
        pred = np.array([[1.5, 2.0]])
        assert abs_rel(pred, target) == depth_metrics(pred, target)["abs_rel"]

    def test_all_invalid_returns_zeros(self) -> None:
        target = np.zeros((2, 2))
        m = depth_metrics(np.ones((2, 2)), target)
        assert m["num_valid"] == 0.0
        assert m["abs_rel"] == 0.0


class TestNormalMetrics:
    def test_identical_normals_zero_error(self) -> None:
        n = np.zeros((2, 2, 3))
        n[..., 2] = 1.0  # all point +z
        m = normal_metrics(n.copy(), n)
        assert m["mean_angular_error"] == 0.0
        assert m["within_11_25"] == 1.0

    def test_orthogonal_normals_90_degrees(self) -> None:
        pred = np.zeros((1, 1, 3))
        pred[..., 0] = 1.0  # +x
        target = np.zeros((1, 1, 3))
        target[..., 1] = 1.0  # +y
        err = angular_error_map(pred, target)
        assert abs(float(err[0, 0]) - 90.0) < 1e-6
        assert normal_metrics(pred, target)["within_30"] == 0.0

    def test_normalizes_non_unit_inputs(self) -> None:
        # Same direction, different magnitudes -> still 0 error after normalizing.
        pred = np.full((1, 1, 3), 5.0)
        target = np.full((1, 1, 3), 0.2)
        assert angular_error(pred, target) < 1e-6

    def test_mask_selects_pixels(self) -> None:
        pred = np.zeros((1, 2, 3))
        pred[..., 2] = 1.0
        target = pred.copy()
        target[0, 1, 2] = -1.0  # second pixel flipped -> 180 deg
        mask = np.array([[True, False]])
        assert normal_metrics(pred, target, mask=mask)["mean_angular_error"] < 1e-6
