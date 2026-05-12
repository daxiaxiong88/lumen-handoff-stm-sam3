from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as nn_functional

from lumen.models.encoder_base import EncoderBase
from lumen.models.registry import register_encoder


class DINOv3Encoder(EncoderBase):
    """Dense patch-token adapter for DINOv3 ViT models from Transformers."""

    def __init__(
        self,
        model_dir: str | Path = "model/dino/dinov3-vits16-pretrain-lvd1689m",
        *,
        device: torch.device | str | None = None,
        local_files_only: bool = True,
        normalize: bool = True,
        auto_convert_input_channels: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "DINOv3Encoder requires transformers. Install with "
                "`uv pip install --python .venv/bin/python transformers modelscope`."
            ) from exc

        target_device = torch.device("cpu" if device is None else device)
        self.model_dir = Path(model_dir)
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_dir,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            local_files_only=local_files_only,
        ).to(target_device)
        self.model.eval()

        self.patch_size = int(self.model.config.patch_size)
        self.in_channels = int(self.model.config.num_channels)
        self.embed_dim = int(self.model.config.hidden_size)
        self.num_register_tokens = int(
            getattr(self.model.config, "num_register_tokens", 0)
        )
        self.normalize = normalize
        self.supports_masked_tokens = False
        self.auto_convert_input_channels = auto_convert_input_channels

        mean = torch.tensor(self.processor.image_mean, dtype=torch.float32).view(
            1, -1, 1, 1
        )
        std = torch.tensor(self.processor.image_std, dtype=torch.float32).view(
            1, -1, 1, 1
        )
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DINOv3 channel adaptation, cropping, and normalization."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4-D input (B, C, H, W), got {x.dim()}-D tensor")
        if self.auto_convert_input_channels and self.in_channels == 3:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[1] == 4:
                x = x[:, :3]
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channel(s), got {x.shape[1]}"
            )

        x = x.float()
        x = self._crop_to_patch_multiple(x)
        if self.normalize:
            mean = self.get_buffer("image_mean").to(x.device)
            std = self.get_buffer("image_std").to(x.device)
            x = (x - mean) / std

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return DINOv3 normalized patch tokens shaped ``(B, N, D)``.

        Gradients flow through this call so the encoder can be fine-tuned.
        Callers that want frozen behavior should call ``.eval()`` and wrap
        the call site in ``torch.inference_mode()`` themselves (see
        :class:`lumen.models.FewShotFeatureMatcher` for an example).
        """
        x = self.preprocess(x)
        outputs = self.model(pixel_values=x)
        first_patch = 1 + self.num_register_tokens
        return outputs.last_hidden_state[:, first_patch:, :]  # type: ignore[no-any-return]

    def token_grid(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """Infer DINOv3's cropped patch-token grid for ``image_size``."""
        height, width = image_size
        return height // self.patch_size, width // self.patch_size

    def _crop_to_patch_multiple(self, x: torch.Tensor) -> torch.Tensor:
        height = (x.shape[-2] // self.patch_size) * self.patch_size
        width = (x.shape[-1] // self.patch_size) * self.patch_size
        if height <= 0 or width <= 0:
            raise ValueError("Input is smaller than one DINOv3 patch")
        return x[..., :height, :width]

    def resize_for_inference(
        self,
        x: torch.Tensor,
        image_size: int | tuple[int, int] = 512,
    ) -> torch.Tensor:
        """Resize CHW/BCHW image tensors before dense inference."""
        size = (image_size, image_size) if isinstance(image_size, int) else image_size
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        out = nn_functional.interpolate(
            x.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        return out.squeeze(0) if squeeze else out


@register_encoder("dinov3")
def load_dinov3_encoder(
    model_dir: str | Path = "model/dino/dinov3-vits16-pretrain-lvd1689m",
    *,
    device: torch.device | str | None = None,
) -> DINOv3Encoder:
    """Load the local DINOv3 ViT-S/16 ModelScope checkpoint."""
    return DINOv3Encoder(model_dir=model_dir, device=device)


__all__ = ["DINOv3Encoder", "load_dinov3_encoder"]
