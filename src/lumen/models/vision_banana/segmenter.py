"""Vision Banana segmenter — FLUX.2-klein-4B as a promptable segmenter.

Reproduces the segmentation behaviour of Vision Banana
(*Image Generators are Generalist Vision Learners*, arXiv:2604.20329) on
top of an open image generator, **black-forest-labs/FLUX.2-klein-4B**
(Apache-2.0). FLUX.2-klein unifies text-to-image and image editing in one
rectified-flow transformer; we prompt it to *generate an RGB segmentation
visualization* of the input and then decode that image back into masks via
:mod:`lumen.models.vision_banana.codecs`.

The model plugs into the Lumen zoo exactly like SAM3 — as a
:class:`~lumen.models.segmenter_base.SegmenterProtocol` registered under
``"vision_banana"`` — so it composes with
:mod:`lumen.data.supervision_bridge` and the existing prelabel loop.

The diffusers import is lazy: ``import lumen.models.vision_banana`` stays
cheap; the heavy ``FLUX.2-klein-4B`` weights only load when the factory is
called. Install the optional extra first::

    uv pip install -e ".[vision_banana]"
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lumen.models.registry import register_segmenter
from lumen.models.segmenter_base import SegmenterBase
from lumen.models.vision_banana.codecs import (
    DEFAULT_BACKGROUND_TOLERANCE,
    DEFAULT_MIN_INSTANCE_AREA,
    DEFAULT_SEMANTIC_TOLERANCE,
    ColorMap,
    build_depth_prompt,
    build_normal_prompt,
    build_segmentation_prompt,
    decode_depth,
    decode_normal,
    decode_segmentation,
    mask_to_xyxy,
    normalize_color_map,
    to_pil_uint8,
)

if TYPE_CHECKING:  # pragma: no cover
    import supervision as sv

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_NUM_INFERENCE_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 1.0


class VisionBananaSegmenter(SegmenterBase):
    """Promptable segmenter backed by a FLUX.2-klein-4B image generator.

    The generator is prompted to emit an RGB segmentation visualization
    (one colour per class, or one colour per instance), which is decoded
    to :class:`supervision.Detections` via the colour codecs.

    Args:
        pipe: An already-loaded diffusers pipeline (the
            ``Flux2KleinPipeline`` returned by
            ``DiffusionPipeline.from_pretrained(DEFAULT_MODEL_ID, ...)``).
        device: Device used to build the RNG generator (only relevant when a
            ``seed`` is passed to :meth:`predict`). Inferred from
            ``torch.cuda`` when omitted.
        num_inference_steps / guidance_scale: Sampling defaults; overridable
            per-call.
    """

    supports_text_prompts = True
    supports_box_prompts = False
    supports_point_prompts = False

    def __init__(
        self,
        pipe: Any,
        *,
        device: torch.device | str | None = None,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        semantic_tolerance: float = DEFAULT_SEMANTIC_TOLERANCE,
        background_tolerance: float = DEFAULT_BACKGROUND_TOLERANCE,
        min_instance_area: int = DEFAULT_MIN_INSTANCE_AREA,
    ) -> None:
        super().__init__()
        self.pipe = pipe
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.semantic_tolerance = semantic_tolerance
        self.background_tolerance = background_tolerance
        self.min_instance_area = min_instance_area
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.image_size = (1024, 1024)

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor | np.ndarray,
        *,
        boxes: torch.Tensor | np.ndarray | None = None,
        points: torch.Tensor | np.ndarray | None = None,
        labels: torch.Tensor | np.ndarray | None = None,
        text: str | list[str] | None = None,
        multimask: bool = False,
        class_colors: MappingLike | None = None,
        instance: bool = False,
        background: object | None = None,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> sv.Detections:
        """Generate an RGB segmentation and decode it to ``sv.Detections``.

        Args:
            image: Input image — ``(H, W)``, ``(H, W, C)``, or ``(C, H, W)``.
            class_colors: ``{class_name: (r, g, b)}`` palette used to build
                the prompt (when *text* is omitted) and to decode the output.
                Required unless *text* carries a full prompt *and* a palette
                is unnecessary (rare); decoding always needs colours.
            instance: If ``True``, request per-instance colours and decode via
                connected components (one mask per instance).
            background: Background colour for instance decoding (defaults to
                the ``"background"`` entry of *class_colors*, else black).
            text: Override prompt. When given, *class_colors* is still used to
                decode the output.
            seed: Optional RNG seed for reproducible generation.
        """
        import supervision as sv

        del boxes, labels, multimask  # accepted for SegmenterProtocol parity
        if points is not None:
            raise NotImplementedError(
                "Vision Banana does not accept point prompts; use text/colour prompts."
            )

        palette: ColorMap = normalize_color_map(class_colors)
        if not palette and text is None:
            raise ValueError(
                "VisionBananaSegmenter.predict needs `class_colors` (and/or "
                "`text`); decoding requires a colour palette."
            )

        prompt = text if isinstance(text, str) else build_segmentation_prompt(
            palette, instance=instance, background=background
        )

        generated = self._generate_rgb(
            image,
            prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )

        bg = background if background is not None else palette.get("background")
        decoded = decode_segmentation(
            generated,
            palette,
            instance=instance,
            background=bg,
            tolerance=self.semantic_tolerance,
        )

        if not decoded:
            return sv.Detections.empty()

        boxes_xyxy: list[list[float]] = []
        masks: list[np.ndarray] = []
        class_ids: list[int] = []
        confidences: list[float] = []
        class_names: list[str] = []
        for class_id, (name, mask) in enumerate(decoded):
            boxes_xyxy.append(list(mask_to_xyxy(mask)))
            masks.append(mask)
            class_ids.append(class_id)
            confidences.append(
                self._mask_confidence(generated, mask, palette, name, instance, bg)
            )
            class_names.append(name)

        detections = sv.Detections(
            xyxy=np.asarray(boxes_xyxy, dtype=np.float32),
            mask=np.stack(masks, axis=0).astype(bool),
            class_id=np.asarray(class_ids, dtype=int),
            confidence=np.asarray(confidences, dtype=np.float32),
        )
        # Attach human-readable names for downstream consumers (prelabel/LS).
        with contextlib.suppress(AttributeError):
            detections.data["class_name"] = np.asarray(class_names, dtype=object)
        return detections

    # ------------------------------------------------------------------
    # Dense-prediction tasks: depth & surface normals
    # ------------------------------------------------------------------

    def _generate_rgb(
        self,
        image: torch.Tensor | np.ndarray,
        prompt: str,
        *,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> np.ndarray:
        """Run the generator and return the RGB output resized to the input."""
        rgb_in = self._to_rgb_uint8(image)
        h, w = int(rgb_in.shape[0]), int(rgb_in.shape[1])
        out_h = int(height) if height is not None else h
        out_w = int(width) if width is not None else w
        generated = self._generate(
            rgb_in,
            prompt,
            out_h,
            out_w,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        if generated.shape[0] != h or generated.shape[1] != w:
            generated = self._resize_nearest(generated, h, w)
        # Expose the raw generated RGB for inspection/debugging (the model's
        # literal output before colour decoding).
        self.last_generated = generated
        return generated

    def predict_depth(
        self,
        image: torch.Tensor | np.ndarray,
        *,
        prompt: str | None = None,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> np.ndarray:
        """Generate a metric-depth visualization and decode it to metres (HxW).

        The model is prompted to emit a rainbow depth image (Vision Banana
        style); :func:`~lumen.models.vision_banana.codecs.decode_depth` inverts
        the power-transform + cube-edge colormap back to metric depth.
        """
        text = prompt if isinstance(prompt, str) else build_depth_prompt()
        generated = self._generate_rgb(
            image,
            text,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )
        return decode_depth(generated)

    def predict_normal(
        self,
        image: torch.Tensor | np.ndarray,
        *,
        prompt: str | None = None,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> np.ndarray:
        """Generate a surface-normal visualization and decode it (HxWx3, unit).

        Returns camera-space unit normals (+x right, +y up, +z toward camera).
        """
        text = prompt if isinstance(prompt, str) else build_normal_prompt()
        generated = self._generate_rgb(
            image,
            text,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )
        return decode_normal(generated)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        rgb_in: np.ndarray,
        prompt: str,
        height: int,
        width: int,
        *,
        seed: int | None,
        num_inference_steps: int | None,
        guidance_scale: float | None,
    ) -> np.ndarray:
        """Run the FLUX.2-klein pipeline and return an ``HxWx3`` uint8 array."""
        pil_in = to_pil_uint8(rgb_in)
        call_kwargs: dict[str, Any] = {
            "image": pil_in,
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": (
                self.num_inference_steps
                if num_inference_steps is None
                else num_inference_steps
            ),
            "guidance_scale": (
                self.guidance_scale if guidance_scale is None else guidance_scale
            ),
        }
        generator = self._make_generator(seed)
        if generator is not None:
            call_kwargs["generator"] = generator

        # Run under bfloat16 autocast. FLUX.2-klein's Qwen3 text encoder can
        # emit float32 hidden states while the transformer weights are
        # bfloat16; autocast reconciles the matmul dtypes so the transformer's
        # context embedder doesn't raise a dtype mismatch.
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.pipe(**call_kwargs)
        return self._output_to_rgb(output)

    def _make_generator(self, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        return torch.Generator(device=self.device).manual_seed(int(seed))

    @staticmethod
    def _output_to_rgb(output: Any) -> np.ndarray:
        images = getattr(output, "images", output)
        first = images[0] if isinstance(images, (list, tuple)) else images
        if hasattr(first, "convert"):  # PIL.Image
            return np.asarray(first.convert("RGB"), dtype=np.uint8)
        arr = np.asarray(first)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            lo, hi = float(arr.min()), float(arr.max())
            arr = (
                np.zeros(arr.shape, dtype=np.uint8)
                if hi <= lo
                else ((arr - lo) / (hi - lo) * 255.0).round().astype(np.uint8)
            )
        return arr.astype(np.uint8)

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------

    def _mask_confidence(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        palette: ColorMap,
        name: str,
        instance: bool,
        background: object | None,
    ) -> float:
        """Colour-match confidence in ``[0, 1]`` for a decoded mask.

        Semantic: mean over-mask of ``clip(1 - dist/tolerance)`` to the
        class's target colour. Instance: ``1.0`` (no single target colour).
        """
        if instance:
            return 1.0
        target = palette.get(name)
        if target is None:
            # instance-style name "foo_3" → fall back to the class palette entry
            base = name.rsplit("_", 1)[0] if "_" in name else name
            target = palette.get(base)
        if target is None:
            return 1.0
        from lumen.models.vision_banana.codecs import color_distance

        dist = color_distance(rgb, target)
        score = np.clip(1.0 - dist / self.semantic_tolerance, 0.0, 1.0)
        return float(score[mask].mean())

    # ------------------------------------------------------------------
    # Image normalisation (mirrors lumen.models.sam3.Sam3Segmenter)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_rgb_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert any common scientific-image layout to ``HxWx3`` uint8."""
        if torch.is_tensor(image):
            arr = image.detach().cpu().numpy()
        else:
            arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            lo, hi = float(arr.min()), float(arr.max())
            if hi <= lo:
                arr = np.zeros(arr.shape, dtype=np.uint8)
            else:
                arr = ((arr - lo) / (hi - lo) * 255.0).round().astype(np.uint8)
        return arr

    @staticmethod
    def _resize_nearest(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
        """Nearest-neighbour resize of an ``HxWx3`` array — preserves colours."""
        pil = to_pil_uint8(rgb)
        return np.asarray(pil.resize((w, h)), dtype=np.uint8)


# A loose structural alias so the type checker accepts plain ``dict``.
MappingLike = Any


@register_segmenter("vision_banana")
def load_vision_banana_segmenter(
    model_id: str | Path = DEFAULT_MODEL_ID,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = None,
    enable_cpu_offload: bool = False,
    pipeline_class: str | None = None,
    download: bool = True,
    **pipe_kwargs: Any,
) -> VisionBananaSegmenter:
    """Load FLUX.2-klein-4B and wrap it as a :class:`VisionBananaSegmenter`.

    Mirrors the canonical diffusers snippet::

        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            dtype=torch.bfloat16, device_map="cuda",
        )

    Args:
        model_id: HuggingFace repo id or local checkpoint dir.
        device: Device hint for the RNG generator only (the pipeline manages
            its own placement via ``device_map`` / offload).
        dtype: Torch dtype for the pipeline (``bfloat16`` by default).
        device_map: e.g. ``"cuda"``. Opt-in: when set, the pipeline is placed
            via accelerate at load time. Defaults to ``None``, in which case the
            pipeline is moved to *device* via ``.to()`` — preferred, because it
            keeps a uniform dtype across all sub-models.
        enable_cpu_offload: Use ``pipe.enable_model_cpu_offload()`` instead of
            ``.to()`` (ignored when *device_map* is set).
        pipeline_class: Optional override — ``"Flux2KleinPipeline"`` forces the
            dedicated class; otherwise the generic ``DiffusionPipeline`` is
            used (it auto-routes to the FLUX.2-klein implementation).
        download: If ``False`` and the checkpoint is missing, raise rather
            than fetch from the Hub.
    """
    try:
        from diffusers import DiffusionPipeline
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Vision Banana support requires `diffusers` (git build for "
            "FLUX.2-klein). Install with `uv pip install -e \".[vision_banana]\"`."
        ) from exc

    target = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    src = str(model_id)
    # Local-only guard: refuse a silent Hub fetch when explicitly asked.
    if (
        not download
        and not Path(src).exists()
        and "/" in src
        and not _is_cached(src)
    ):
        raise FileNotFoundError(
            f"Checkpoint {src!r} not found locally and download=False."
        )

    # Use ``torch_dtype`` (not ``dtype``): the dedicated Flux2KleinPipeline
    # rejects ``dtype=`` and would otherwise warn + silently load float32.
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype, **pipe_kwargs}
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    if pipeline_class == "DiffusionPipeline":
        pipe = DiffusionPipeline.from_pretrained(src, **load_kwargs)
    else:
        try:
            from diffusers import Flux2KleinPipeline

            pipe = Flux2KleinPipeline.from_pretrained(src, **load_kwargs)
        except ImportError:
            # Older diffusers without Flux2KleinPipeline — generic routing
            # still resolves to the FLUX.2-klein implementation.
            pipe = DiffusionPipeline.from_pretrained(src, **load_kwargs)

    if device_map is None:
        # `.to(device)` keeps a uniform dtype across every sub-model
        # (including the text encoder). Loading with ``device_map="cuda"``
        # instead leaves the Qwen3 text encoder in float32 while the
        # transformer is bfloat16, which raises a dtype error inside the
        # transformer's context embedder. Opt into ``device_map``/offload
        # explicitly only when you need them.
        if enable_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(target)

    return VisionBananaSegmenter(pipe, device=target)


def _is_cached(model_id: str) -> bool:
    """Best-effort check that a HF repo is present in the local hub cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        return try_to_load_from_cache(model_id, "config.json") is not None
    except Exception:
        return False


__all__ = [
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_MODEL_ID",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "VisionBananaSegmenter",
    "load_vision_banana_segmenter",
]
