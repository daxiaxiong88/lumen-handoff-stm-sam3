"""Phase 2 — LoRA instruction-tuning for the Vision Banana reproduction.

Vision Banana's quality comes from **instruction-tuning** the image generator on
vision-task data formatted as RGB outputs (the paper mixes it in at a low ratio).
This module provides a LoRA fine-tuner that teaches FLUX.2-klein-4B to emit clean,
flat-colour segmentation masks given (input image + segmentation prompt), so the
zero-shot decoder (:mod:`lumen.models.vision_banana.codecs`) recovers good masks.

Training objective: **rectified-flow (flow-matching)** matching the Flux2Klein
inference path. The target segmentation image and the input image are both
VAE-encoded + patchified + batch-norm normalised via the pipeline's own
``prepare_image_latents``; the noisy target is concatenated with the input-image
latents along the sequence axis (exactly how the pipeline conditions the
transformer), and the model predicts the flow velocity.

Only the transformer is LoRA-adapted (PEFT); the VAE and text encoder are frozen.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import torch
import torch.nn.functional as nn_functional

from lumen.models.vision_banana.codecs import (
    ColorMap,
    build_depth_prompt,
    build_normal_prompt,
    build_segmentation_prompt,
    encode_depth,
    encode_normal,
    encode_segmentation,
    parse_color,
)

logger = logging.getLogger(__name__)

# Flux2Transformer2DModel attention projections (verified against the cached
# weights). The ``add_*_proj`` linears feed the reference/input-image tokens, so
# targeting them lets LoRA learn image-conditioned segmentation.
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "to_qkv_mlp_proj",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
)


@dataclass
class LoRAConfig:
    """PEFT LoRA hyper-parameters for Flux2Klein instruction-tuning."""

    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    bias: str = "none"
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS
    lr: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 0


# ---------------------------------------------------------------------------
# Model-independent helpers (unit-tested without FLUX.2-klein)
# ---------------------------------------------------------------------------


def to_output_latent_ids(reference_ids: torch.Tensor) -> torch.Tensor:
    """Convert reference-image position IDs (T=scale) to output-latent IDs (T=0).

    The Vision Banana training target is the *denoised* latent. At inference the
    Flux2Klein pipeline stamps the denoised latent with the output temporal
    position ``T=0`` (``prepare_latents`` -> ``_prepare_latent_ids``), while the
    conditioning image gets ``T=10`` (``prepare_image_latents`` ->
    ``_prepare_image_ids``, ``scale=10``).

    The trainer builds the target latent via ``prepare_image_latents`` for
    convenience, which wrongly stamps it ``T=10`` too — so target tokens collide
    with the conditioning tokens' positions and training diverges from the ``T=0``
    inference path. Zeroing the T column (col 0) yields exactly the grid
    ``_prepare_latent_ids`` would produce (identical H/W/L, ``T=0``), realigning
    training with inference.
    """
    ids = reference_ids.clone()
    ids[..., 0] = 0
    return ids


def make_segmentation_target(
    label_map: np.ndarray,
    class_names: Sequence[str],
    class_colors: ColorMap,
    *,
    instance: bool = False,
) -> tuple[str, np.ndarray]:
    """Build a (prompt, target_rgb) training pair from a label map.

    The target is a flat RGB segmentation image (:func:`encode_segmentation`)
    — the exact signal Vision Banana trains on — and the prompt is the matching
    Vision-Banana-style instruction.

    Args:
        label_map: ``(H, W)`` int array; value ``i`` → ``class_names[i]``.
        class_names: Ordered class names (``class_names[i]`` ↔ label ``i``).
        class_colors: ``{class_name: (r, g, b)}`` palette.
        instance: Emit an instance-style prompt (target is still per-class).
    """
    if label_map.ndim != 2:
        raise ValueError(f"Expected a 2-D label map, got shape {label_map.shape}")
    palette = [parse_color(class_colors[name]) for name in class_names]
    target_rgb = encode_segmentation(label_map, palette)
    prompt = build_segmentation_prompt(class_colors, instance=instance)
    return prompt, target_rgb


def make_depth_target(depth_map: np.ndarray, *, prompt: str | None = None) -> tuple[str, np.ndarray]:
    """Build a (prompt, target_rgb) pair for metric-depth instruction-tuning.

    ``depth_map`` is HxW metres (≥0); the target is the rainbow cube-edge
    depth visualization (:func:`encode_depth`) — Vision Banana's depth signal.
    """
    return prompt or build_depth_prompt(), encode_depth(depth_map)


def make_normal_target(normal_map: np.ndarray, *, prompt: str | None = None) -> tuple[str, np.ndarray]:
    """Build a (prompt, target_rgb) pair for surface-normal instruction-tuning.

    ``normal_map`` is HxWx3 camera-space (components in [−1, 1]).
    """
    return prompt or build_normal_prompt(), encode_normal(normal_map)


def flow_match(
    target: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectified-flow interpolation and velocity target.

    Matches ``FlowMatchEulerDiscreteScheduler.scale_noise`` exactly:
    ``x_sigma = σ·noise + (1−σ)·data`` (so σ=1 → noise, σ=0 → data), and the
    scheduler's Euler update ``prev = x + (σ_next − σ)·model_output`` implies the
    model regresses ``model_output = dx/dσ = noise − data``.

    ``sigma`` is broadcastable to ``target`` (e.g. shape ``(B, 1, 1)``).

    Returns ``(noisy, velocity)`` with ``velocity = noise − target``.
    """
    noisy = sigma * noise + (1.0 - sigma) * target
    velocity = noise - target
    return noisy, velocity


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class SegmentationDataset:
    """Yields ``(image, label_map, class_names, class_colors)`` training samples.

    Subclass and override :meth:`__getitem__` / :meth:`__len__` to feed a real
    dataset (Cityscapes, ADE20k, microscopy, …). :meth:`sample` turns each item
    into a ``(prompt, target_rgb, image)`` triple via
    :func:`make_segmentation_target`.
    """

    def __init__(
        self,
        class_names: Sequence[str],
        class_colors: ColorMap,
        *,
        instance: bool = False,
    ) -> None:
        self.class_names = list(class_names)
        self.class_colors = dict(class_colors)
        self.instance = instance

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(image_HxWx3_uint8, label_map_HxW_int)`` for sample *idx*."""
        raise NotImplementedError

    def sample(self, idx: int) -> tuple[np.ndarray, str, np.ndarray]:
        image, label_map = self[idx]
        prompt, target_rgb = make_segmentation_target(
            label_map, self.class_names, self.class_colors, instance=self.instance
        )
        return image, prompt, target_rgb


class InMemorySegDataset(SegmentationDataset):
    """Simple in-memory dataset (images + label maps) — used for tests/smoke."""

    def __init__(
        self,
        images: Sequence[np.ndarray],
        label_maps: Sequence[np.ndarray],
        class_names: Sequence[str],
        class_colors: ColorMap,
        *,
        instance: bool = False,
    ) -> None:
        super().__init__(class_names, class_colors, instance=instance)
        if len(images) != len(label_maps):
            raise ValueError("images and label_maps must have equal length")
        self.images = list(images)
        self.label_maps = list(label_maps)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.images[idx], self.label_maps[idx]


class PairDataset:
    """Generic in-memory dataset for dense-prediction tasks (depth, normals, …).

    Holds parallel ``(image, dense_map)`` arrays and a *target_builder* callable
    that turns a ``dense_map`` into a ``(prompt, target_rgb)`` pair — e.g.
    :func:`make_depth_target` or :func:`make_normal_target`.
    """

    def __init__(
        self,
        images: Sequence[np.ndarray],
        maps: Sequence[np.ndarray],
        target_builder: Callable[[np.ndarray], tuple[str, np.ndarray]],
    ) -> None:
        if len(images) != len(maps):
            raise ValueError("images and maps must have equal length")
        self.images = list(images)
        self.maps = list(maps)
        self.target_builder = target_builder

    def __len__(self) -> int:
        return len(self.images)

    def sample(self, idx: int) -> tuple[np.ndarray, str, np.ndarray]:
        prompt, target_rgb = self.target_builder(self.maps[idx])
        return self.images[idx], prompt, target_rgb


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Flux2KleinLoRATrainer:
    """LoRA instruction-tuner for FLUX.2-klein-4B segmentation.

    Wraps a loaded ``Flux2KleinPipeline`` (from
    :func:`~lumen.models.vision_banana.segmenter.load_vision_banana_segmenter`
    or ``diffusers``): freezes the VAE + text encoder, attaches LoRA adapters to
    the transformer, and runs rectified-flow training on RGB segmentation targets.

    Example::

        from lumen.models import build_segmenter
        from lumen.training.generative import Flux2KleinLoRATrainer, LoRAConfig, InMemorySegDataset

        seg = build_segmenter("vision_banana")
        trainer = Flux2KleinLoRATrainer(seg.pipe, LoRAConfig(rank=16))
        losses = trainer.fit(dataset, steps=200, height=512, width=512)
        trainer.save_lora("vision_banana_seg_lora")
    """

    def __init__(
        self,
        pipe: Any,
        config: LoRAConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.pipe = pipe
        self.config = config or LoRAConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.gen = torch.Generator(device=self.device).manual_seed(self.config.seed)
        self._setup()

    # -- setup -----------------------------------------------------------

    def _setup(self) -> None:
        pipe = self.pipe
        pipe.vae.eval()
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.eval()
        pipe.text_encoder.requires_grad_(False)
        pipe.transformer.requires_grad_(False)

        from peft import LoraConfig as _PeftLoraConfig
        from peft import get_peft_model

        cfg = self.config
        peft_config = _PeftLoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            bias=cfg.bias,
            target_modules=list(cfg.target_modules),
        )
        pipe.transformer = get_peft_model(pipe.transformer, peft_config)
        pipe.transformer.train()
        pipe.transformer.to(self.dtype)

        trainable = [p for p in pipe.transformer.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "LoRA attached no trainable parameters — `target_modules` "
                f"({cfg.target_modules}) match no transformer linears."
            )
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        n_train = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in pipe.transformer.parameters())
        logger.info(
            "LoRA: %d / %d params trainable (%.2f%%)",
            n_train,
            n_total,
            100.0 * n_train / max(n_total, 1),
        )

    # -- sample preparation ---------------------------------------------

    def _to_input_tensor(
        self, image: np.ndarray, height: int, width: int
    ) -> torch.Tensor:
        """Preprocess an HxWx3 uint8 image to an NCHW tensor for the VAE."""
        from PIL import Image

        pil = Image.fromarray(np.asarray(image).astype(np.uint8))
        return cast(
            torch.Tensor,
            self.pipe.image_processor.preprocess(pil, height=height, width=width),
        )

    def prepare_sample(
        self,
        image: np.ndarray,
        prompt: str,
        target_rgb: np.ndarray,
        *,
        height: int,
        width: int,
    ) -> dict[str, torch.Tensor]:
        """Encode a (image, prompt, target) triple into transformer inputs."""
        pipe = self.pipe
        device = self.device

        target_t = self._to_input_tensor(target_rgb, height, width)
        image_t = self._to_input_tensor(image, height, width)

        prompt_embeds, text_ids = pipe.encode_prompt(prompt, device=device)
        target_latents, latent_ids = pipe.prepare_image_latents(
            images=[target_t], batch_size=1, generator=self.gen,
            device=device, dtype=pipe.vae.dtype,
        )
        # The target is the denoised/output latent: stamp it with the output
        # temporal position (T=0) that inference uses, not the reference-image
        # position (T=10) prepare_image_latents assigns. See to_output_latent_ids.
        latent_ids = to_output_latent_ids(latent_ids)
        image_latents, image_latent_ids = pipe.prepare_image_latents(
            images=[image_t], batch_size=1, generator=self.gen,
            device=device, dtype=pipe.vae.dtype,
        )
        return {
            "target_latents": target_latents,
            "latent_ids": latent_ids,
            "image_latents": image_latents,
            "image_latent_ids": image_latent_ids,
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
        }

    # -- training --------------------------------------------------------

    def train_step(self, sample: dict[str, torch.Tensor]) -> float:
        """One rectified-flow optimisation step; returns the scalar loss."""
        pipe = self.pipe
        device = self.device
        # Derive dtype from a parameter — a PEFT-wrapped transformer may not
        # expose ``.dtype`` directly.
        tdtype = next(pipe.transformer.parameters()).dtype

        target_latents = sample["target_latents"].to(device)
        image_latents = sample["image_latents"].to(device)
        latent_ids = sample["latent_ids"].to(device)
        image_latent_ids = sample["image_latent_ids"].to(device)
        prompt_embeds = sample["prompt_embeds"].to(device, tdtype)
        text_ids = sample["text_ids"].to(device)

        b = target_latents.shape[0]
        noise = torch.randn(
            target_latents.shape, device=device, dtype=target_latents.dtype,
            generator=self.gen,
        )
        sigma = torch.rand((b,), device=device, dtype=tdtype, generator=self.gen)
        sigma_b = sigma.reshape(b, 1, 1)
        noisy, velocity = flow_match(target_latents, noise, sigma_b)
        noisy = noisy.to(tdtype)
        velocity = velocity.to(tdtype)

        hidden = torch.cat([noisy, image_latents.to(tdtype)], dim=1)
        img_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16,
            enabled=(self.device.type == "cuda"),
        ):
            pred = pipe.transformer(
                hidden_states=hidden,
                timestep=sigma,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
            pred = pred[:, : target_latents.shape[1]]
            loss = nn_functional.mse_loss(pred.float(), velocity.float())

        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def fit(
        self,
        dataset: SegmentationDataset,
        *,
        steps: int,
        height: int = 512,
        width: int = 512,
        log_every: int = 0,
        log_fn: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        """Train for *steps* optimisation steps; return the per-step losses."""
        losses: list[float] = []
        n = len(dataset)
        if n == 0:
            raise ValueError("dataset is empty")
        for step in range(steps):
            image, prompt, target_rgb = dataset.sample(step % n)
            sample = self.prepare_sample(
                image, prompt, target_rgb, height=height, width=width
            )
            loss = self.train_step(sample)
            losses.append(loss)
            if log_fn is not None or log_every:
                every = log_every or 1
                if step % every == 0:
                    if log_fn is not None:
                        log_fn(step, loss)
                    else:
                        logger.info("step %d  loss=%.5f", step, loss)
        return losses

    # -- persistence -----------------------------------------------------

    def save_lora(self, path: str | Path) -> None:
        """Save the LoRA adapter (PEFT format) to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.pipe.transformer.save_pretrained(path)
        logger.info("Saved LoRA adapter to %s", path)

    @classmethod
    def load_for_inference(
        cls,
        lora_path: str | Path,
        model_id: str = "black-forest-labs/FLUX.2-klein-4B",
        **loader_kwargs: Any,
    ) -> Any:
        """Load FLUX.2-klein with a trained LoRA adapter and return a segmenter.

        Mirrors :func:`load_vision_banana_segmenter` but attaches the saved
        adapter to the transformer before wrapping it as a segmenter.
        """
        from lumen.models.vision_banana.segmenter import (
            VisionBananaSegmenter,
            load_vision_banana_segmenter,
        )

        seg = cast(
            VisionBananaSegmenter,
            load_vision_banana_segmenter(model_id, **loader_kwargs),
        )
        from peft import PeftModel

        seg.pipe.transformer = PeftModel.from_pretrained(
            seg.pipe.transformer, str(lora_path)
        )
        seg.pipe.transformer.eval()
        return seg


__all__ = [
    "DEFAULT_LORA_TARGETS",
    "Flux2KleinLoRATrainer",
    "InMemorySegDataset",
    "LoRAConfig",
    "PairDataset",
    "SegmentationDataset",
    "flow_match",
    "make_depth_target",
    "make_normal_target",
    "make_segmentation_target",
    "to_output_latent_ids",
]
