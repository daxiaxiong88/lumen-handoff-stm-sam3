"""Tests for the in-memory STM mixed-annotation canonicalization path."""

from __future__ import annotations

import numpy as np

from lumen.annotation.stm_canonical import (
    DEFAULT_STM_CLASS_NAMES,
    CanonicalShape,
    canonicalize_label_studio_regions,
    canonicalize_labelme_shapes,
    evaluate_dot_instances,
    semantic_iou_summary,
)


def test_defect_priority_overwrites_overlapping_modulation() -> None:
    """A defect disk must remain a defect where it overlaps modulation."""
    shapes = [
        {
            "label": "dark_defect",
            "shape_type": "point",
            "points": [[50, 50]],
        },
        {
            "label": "sqrt2_modulation_region",
            "shape_type": "polygon",
            "points": [[20, 20], [80, 20], [80, 80], [20, 80]],
        },
    ]

    label_map, _ = canonicalize_labelme_shapes(
        shapes,
        image_size=(100, 100),
        crop_top=0,
        target_size=(100, 100),
        point_radii={"dark_defect": 5},
    )

    assert label_map[50, 50] == DEFAULT_STM_CLASS_NAMES.index("dark_defect")
    assert label_map[30, 30] == DEFAULT_STM_CLASS_NAMES.index("sqrt2_modulation_region")


def test_point_rectangle_and_polygon_share_crop_resize_coordinates() -> None:
    """All three LabelMe geometries map into the same cropped output frame."""
    shapes = [
        {
            "label": "dark_defect",
            "shape_type": "point",
            "points": [[20, 30]],
        },
        {
            "label": "bright_defect",
            "shape_type": "rectangle",
            "points": [[40, 30], [60, 50]],
        },
        {
            "label": "modulation_region",
            "shape_type": "polygon",
            "points": [[70, 30], [90, 30], [90, 50], [70, 50]],
        },
    ]

    label_map, _ = canonicalize_labelme_shapes(
        shapes,
        image_size=(100, 110),
        crop_top=10,
        target_size=(100, 100),
        point_radii={"dark_defect": 3},
    )

    assert label_map[20, 20] == DEFAULT_STM_CLASS_NAMES.index("dark_defect")
    assert label_map[30, 50] == DEFAULT_STM_CLASS_NAMES.index("bright_defect")
    assert label_map[30, 80] == DEFAULT_STM_CLASS_NAMES.index("modulation_region")


def test_label_studio_percent_polygons_apply_the_same_class_priority() -> None:
    regions = [
        {
            "value": {
                "polygonlabels": ["bright_defect"],
                "points": [[50, 50], [60, 50], [60, 60], [50, 60]],
            }
        },
        {
            "value": {
                "polygonlabels": ["modulation_region"],
                "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
            }
        },
    ]

    label_map, _ = canonicalize_label_studio_regions(
        regions,
        image_size=(100, 100),
        crop_top=0,
        target_size=(100, 100),
    )

    assert label_map[55, 55] == DEFAULT_STM_CLASS_NAMES.index("bright_defect")
    assert label_map[10, 10] == DEFAULT_STM_CLASS_NAMES.index("modulation_region")


def test_semantic_iou_is_pooled_over_images_not_averaged_per_image() -> None:
    first_gt = np.zeros((2, 2), dtype=np.int32)
    first_gt[0, 0] = 1
    first_pred = first_gt.copy()
    second_gt = np.zeros((2, 2), dtype=np.int32)
    second_gt[0, 0] = 1
    second_pred = np.zeros((2, 2), dtype=np.int32)

    report = semantic_iou_summary(
        [first_pred, second_pred],
        [first_gt, second_gt],
        class_names=["background", "dot"],
    )

    # pooled: one correct foreground pixel / two foreground pixels
    assert report["per_class_iou"]["dot"] == 0.5
    assert report["foreground_miou"] == 0.5


def test_dot_metrics_match_one_prediction_to_one_annotation() -> None:
    gt_shapes = [
        CanonicalShape("dark_defect", "point", ((10.0, 10.0),)),
        CanonicalShape("dark_defect", "point", ((30.0, 10.0),)),
    ]
    pred = np.zeros((40, 40), dtype=np.int32)
    pred[9:12, 9:12] = 1
    pred[9:12, 27:30] = 1
    pred[30:33, 30:33] = 1

    report = evaluate_dot_instances(
        pred,
        gt_shapes,
        class_names=["background", "dark_defect"],
        tolerance_by_class={"dark_defect": 4.0},
    )

    metrics = report["per_class"]["dark_defect"]
    assert metrics == {
        "tp": 2,
        "fp": 1,
        "fn": 0,
        "precision": 2 / 3,
        "recall": 1.0,
        "f1": 0.8,
    }
