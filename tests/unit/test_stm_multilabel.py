from __future__ import annotations

import numpy as np

from lumen.annotation.stm_canonical import CanonicalShape
from lumen.annotation.stm_multilabel import (
    STM_MULTILABEL_CLASSES,
    multilabel_iou_summary,
    rasterize_multilabel_shapes,
)


def test_overlapping_defect_and_modulation_are_both_preserved() -> None:
    shapes = (
        CanonicalShape(
            "modulation_region", "polygon", ((2.0, 2.0), (14.0, 2.0), (14.0, 14.0))
        ),
        CanonicalShape("bright_defect", "point", ((8.0, 8.0),)),
    )
    targets = rasterize_multilabel_shapes(
        shapes,
        target_size=(16, 16),
        point_radii=dict.fromkeys(STM_MULTILABEL_CLASSES, 2),
    )
    bright = STM_MULTILABEL_CLASSES.index("bright_defect")
    modulation = STM_MULTILABEL_CLASSES.index("modulation_region")
    assert targets[bright, 8, 8] == 1
    assert targets[modulation, 8, 8] == 1


def test_multilabel_iou_is_pooled_per_channel() -> None:
    target = np.zeros((4, 4, 4), dtype=bool)
    prediction = np.zeros_like(target)
    target[0, 0:2, 0:2] = True
    prediction[0, 0:2, 0:2] = True
    target[2, 2:4, 2:4] = True
    prediction[2, 2:3, 2:4] = True

    report = multilabel_iou_summary([prediction], [target])

    assert report["per_class_iou"]["dark_defect"] == 1.0
    assert report["per_class_iou"]["modulation_region"] == 0.5
