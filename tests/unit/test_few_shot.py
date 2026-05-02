from __future__ import annotations

import torch

from lumen.models import FewShotFeatureMatcher


class FakePatchEncoder(torch.nn.Module):
    patch_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.avg_pool2d(x[:, :1], kernel_size=2, stride=2)
        f0 = pooled.flatten(2).transpose(1, 2)
        f1 = 1.0 - f0
        return torch.cat([f0, f1], dim=-1)


def test_few_shot_matcher_fits_and_predicts() -> None:
    encoder = FakePatchEncoder()
    matcher = FewShotFeatureMatcher(encoder, min_patch_fraction=0.5)
    image = torch.tensor(
        [
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        ]
    )
    mask = torch.tensor(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ]
    )
    matcher.fit([image], [mask])

    pred = matcher.predict(image, include_pca=False)
    assert set(pred.label_values.tolist()) == {1, 2}
    assert pred.labels.shape == (4, 4)
    assert torch.all(pred.labels[:, :2] == 1)
    assert torch.all(pred.labels[:, 2:] == 2)
    assert pred.confidence.shape == (4, 4)


def test_few_shot_state_roundtrip_and_colorize() -> None:
    encoder = FakePatchEncoder()
    matcher = FewShotFeatureMatcher(encoder)
    image = torch.ones(1, 4, 4)
    mask = torch.ones(4, 4, dtype=torch.long) * 3
    matcher.fit([image], [mask])

    clone = FewShotFeatureMatcher(encoder)
    clone.load_state_dict(matcher.state_dict())
    pred = clone.predict(image, include_pca=False)
    rgb = clone.colorize_labels(pred.labels)

    assert pred.label_values.tolist() == [3]
    assert rgb.shape == (3, 4, 4)
    assert float(rgb.max()) <= 1.0
