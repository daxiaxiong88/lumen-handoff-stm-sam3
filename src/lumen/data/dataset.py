"""Scientific image datasets for the Lumen framework.

Loaders are designed for the file zoo encountered in microscopy
workflows:

- ``.tif`` / ``.tiff`` — generic high-bit-depth image format (handled by
  :mod:`tifffile` when available, falling back to :mod:`PIL`).
- ``.dm3`` / ``.dm4`` — DigitalMicrograph (Gatan) STEM/TEM/EELS files.
- ``.ser`` — FEI/TIA scanning data.
- ``.png`` / ``.jpg`` / ``.jpeg`` — standard image files used for
  references, masks, and labels.

Hyperspy is used for ``.dm3``/``.dm4``/``.ser`` because it is the de
facto reader in the electron-microscopy community and surfaces the
metadata (pixel size, voltage, magnification) that downstream
calibration requires. It is imported lazily so that the dataset can be
constructed in an environment without hyperspy installed; only loading a
file in those formats triggers the import.

Loading is lazy: file paths are scanned at construction time but pixels
are read in :meth:`__getitem__`. Optional metadata extraction is
controlled per-call so that callers can avoid the I/O cost when they
only need pixels.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import torch
import torch.nn.functional as nn_functional
from torch.utils.data import Dataset

PathLike = Union[str, os.PathLike]

SUPPORTED_EXTENSIONS = (
    ".tif",
    ".tiff",
    ".dm3",
    ".dm4",
    ".ser",
    ".png",
    ".jpg",
    ".jpeg",
)


@dataclass
class ImageMetadata:
    """Container for metadata extracted from a scientific image file.

    Attributes:
        path: Source file path.
        shape: Image shape after loading, ``(H, W)`` for single-frame
            data or ``(N, H, W)`` for multi-frame.
        dtype: Numpy dtype string of the original data.
        pixel_size_nm: Pixel size in nanometres if known. ``None`` for
            non-metadata-bearing formats (e.g. PNG).
        voltage_kv: Beam voltage in kilovolts, if known.
        magnification: Reported magnification, if known.
        extras: Format-specific extra metadata dict.
    """

    path: str
    shape: tuple[int, ...]
    dtype: str
    pixel_size_nm: float | None = None
    voltage_kv: float | None = None
    magnification: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _list_images(
    root: PathLike,
    extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS,
    recursive: bool = True,
) -> list[Path]:
    """Walk ``root`` and return matching image paths in stable order."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root_path}")
    if recursive:
        candidates: list[Path] = []
        for ext in extensions:
            candidates.extend(root_path.rglob(f"*{ext}"))
            candidates.extend(root_path.rglob(f"*{ext.upper()}"))
    else:
        candidates = [
            p
            for p in root_path.iterdir()
            if p.suffix.lower() in extensions and p.is_file()
        ]
    return sorted({p.resolve() for p in candidates})


def load_image_array(path: PathLike) -> tuple[np.ndarray, dict[str, Any]]:
    """Read pixels and lightweight metadata from a single file.

    Args:
        path: File path. Extension determines the loader.

    Returns:
        ``(array, metadata)`` where ``array`` is at least 2-D numpy and
        ``metadata`` is a possibly-empty extras dict from the loader.
        For multi-frame files (e.g. some DM3 stacks), ``array`` is
        ``(N, H, W)``.

    Raises:
        ValueError: If the extension is unsupported.
        ImportError: If the file format requires an optional dependency
            that is not installed.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        return _load_tiff(path)
    if ext in (".dm3", ".dm4", ".ser"):
        return _load_hyperspy(path)
    if ext in (".png", ".jpg", ".jpeg"):
        return _load_pil(path)
    raise ValueError(
        f"Unsupported image extension: {ext!r}. " f"Supported: {SUPPORTED_EXTENSIONS}"
    )


def _load_tiff(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a TIFF using :mod:`tifffile` if available, else :mod:`PIL`."""
    try:
        import tifffile

        arr = tifffile.imread(str(path))
        return np.asarray(arr), {"loader": "tifffile"}
    except ImportError:
        from PIL import Image

        with Image.open(path) as im:
            arr = np.asarray(im)
        return arr, {"loader": "pillow"}


def _load_pil(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load PNG/JPEG via :mod:`PIL`."""
    from PIL import Image

    with Image.open(path) as im:
        if im.mode not in ("L", "I", "I;16", "RGB", "RGBA"):
            im = im.convert("L")
        arr = np.asarray(im)
    return arr, {"loader": "pillow", "mode": str(arr.dtype)}


def _load_hyperspy(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a DigitalMicrograph / TIA file via :mod:`hyperspy`."""
    try:
        import hyperspy.api as hs
    except ImportError as exc:
        raise ImportError(
            f"hyperspy is required to read {path.suffix} files; "
            "install with `pip install hyperspy`"
        ) from exc

    signal = hs.load(str(path))
    arr = np.asarray(signal.data)
    metadata = signal.metadata.as_dictionary() if hasattr(signal, "metadata") else {}
    axes = (
        signal.axes_manager.as_dictionary() if hasattr(signal, "axes_manager") else {}
    )
    return arr, {"loader": "hyperspy", "metadata": metadata, "axes": axes}


def _normalize_to_float(arr: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize to ``[0, 1]`` float32, robust to constant images."""
    arr = arr.astype(np.float32, copy=False)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _to_chw_tensor(arr: np.ndarray, frame_index: int | None) -> torch.Tensor:
    """Reduce a possibly-multi-frame array to ``(C, H, W)`` float32 torch."""
    if arr.ndim == 2:
        chw = arr[None, ...]
    elif arr.ndim == 3:
        # Could be (N, H, W) (stack) or (H, W, C) (color image).
        if arr.shape[-1] in (3, 4) and arr.shape[0] not in (3, 4):
            chw = np.moveaxis(arr, -1, 0)
        else:
            idx = 0 if frame_index is None else frame_index
            if idx >= arr.shape[0]:
                raise IndexError(
                    f"frame_index {idx} out of range for stack of {arr.shape[0]} frames"
                )
            chw = arr[idx][None, ...]
    elif arr.ndim == 4:
        # (N, C, H, W) — take frame[0] by default.
        idx = 0 if frame_index is None else frame_index
        if idx >= arr.shape[0]:
            raise IndexError(
                f"frame_index {idx} out of range for stack of {arr.shape[0]} frames"
            )
        chw = arr[idx]
    else:
        raise ValueError(f"Unsupported array rank for image: {arr.ndim}")
    return torch.from_numpy(np.ascontiguousarray(chw)).float()


def ensure_channel_count(tensor: torch.Tensor, channels: int = 1) -> torch.Tensor:
    """Convert a CHW image tensor to a stable channel count.

    Scientific image workflows often mix grayscale PNGs with RGB/RGBA exports.
    This helper makes batching deterministic by converting RGB/RGBA to grayscale
    when ``channels=1`` or by repeating grayscale inputs when ``channels=3``.
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}")
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if tensor.shape[0] == channels:
        return tensor
    if tensor.shape[0] == 4:
        tensor = tensor[:3]
    if channels == 1:
        if tensor.shape[0] == 1:
            return tensor
        if tensor.shape[0] == 3:
            weights = tensor.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
            return (tensor * weights).sum(dim=0, keepdim=True)
    if channels == 3 and tensor.shape[0] == 1:
        return tensor.repeat(3, 1, 1)
    raise ValueError(
        f"Cannot convert image with {tensor.shape[0]} channel(s) to {channels}"
    )


def resize_chw(
    tensor: torch.Tensor,
    image_size: int | tuple[int, int],
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize a CHW tensor with torch interpolation."""
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    is_mask = mode == "nearest"
    x = tensor.unsqueeze(0).float()
    if is_mask:
        out = nn_functional.interpolate(x, size=size, mode=mode)
    else:
        out = nn_functional.interpolate(x, size=size, mode=mode, align_corners=False)
    return out.squeeze(0)


def _extract_metadata(
    path: Path,
    arr: np.ndarray,
    extras: dict[str, Any],
) -> ImageMetadata:
    """Pull pixel size / voltage / magnification out of loader extras."""
    pixel_size_nm: float | None = None
    voltage_kv: float | None = None
    magnification: float | None = None

    if extras.get("loader") == "hyperspy":
        axes = extras.get("axes") or {}
        for axis in axes.values():
            if not isinstance(axis, dict):
                continue
            units = (axis.get("units") or "").lower()
            scale = axis.get("scale")
            if scale is None:
                continue
            if units in ("nm", "nanometer", "nanometre"):
                pixel_size_nm = float(scale)
                break
            if units in ("um", "µm", "micron", "micrometer"):
                pixel_size_nm = float(scale) * 1000.0
                break
            if units in ("a", "å", "angstrom"):
                pixel_size_nm = float(scale) / 10.0
                break
        meta = extras.get("metadata") or {}
        acq = meta.get("Acquisition_instrument", {}) if isinstance(meta, dict) else {}
        for instrument in ("TEM", "SEM"):
            beam = acq.get(instrument, {}) if isinstance(acq, dict) else {}
            if "beam_energy" in beam:
                voltage_kv = float(beam["beam_energy"])
            if "magnification" in beam:
                magnification = float(beam["magnification"])
            if voltage_kv is not None or magnification is not None:
                break

    return ImageMetadata(
        path=str(path),
        shape=tuple(arr.shape),
        dtype=str(arr.dtype),
        pixel_size_nm=pixel_size_nm,
        voltage_kv=voltage_kv,
        magnification=magnification,
        extras=extras,
    )


class ScientificImageDataset(Dataset[dict[str, Any]]):
    """Generic lazy-loading scientific image dataset.

    Walks a root directory at construction time and stores file paths.
    Pixels are read on demand in :meth:`__getitem__`.

    Args:
        root: Root directory containing image files.
        extensions: Tuple of file suffixes (lower-case, with leading dot)
            to include.
        recursive: Whether to walk subdirectories.
        transform: Optional callable applied to the ``(C, H, W)`` torch
            tensor before returning. Useful for augmentation pipelines.
        normalize: If ``True``, scales pixel values to ``[0, 1]`` per
            image before tensor conversion.
        return_metadata: Whether to attach an :class:`ImageMetadata`
            instance to each returned sample.
        frame_index: For multi-frame stacks, which frame to return. Use
            ``None`` to default to frame 0.

    Returns:
        Each sample is a dict with key ``"image"`` (a ``(C, H, W)``
        float tensor) and optionally ``"metadata"`` and ``"path"``.
    """

    def __init__(
        self,
        root: PathLike,
        *,
        extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS,
        recursive: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        normalize: bool = True,
        return_metadata: bool = False,
        frame_index: int | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.extensions = tuple(e.lower() for e in extensions)
        self.recursive = recursive
        self.transform = transform
        self.normalize = normalize
        self.return_metadata = return_metadata
        self.frame_index = frame_index
        self.paths: list[Path] = _list_images(
            self.root, self.extensions, recursive=recursive
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        arr, extras = load_image_array(path)
        if self.normalize:
            arr = _normalize_to_float(arr)
        tensor = _to_chw_tensor(arr, self.frame_index)
        if self.transform is not None:
            tensor = self.transform(tensor)

        sample: dict[str, Any] = {"image": tensor, "path": str(path)}
        if self.return_metadata:
            sample["metadata"] = _extract_metadata(path, arr, extras)
        return sample


class STEMDataset(ScientificImageDataset):
    """Scientific image dataset specialized for STEM data.

    Convenience subclass that defaults to DigitalMicrograph / TIFF
    extensions and always returns metadata (pixel size and voltage are
    typically required to interpret STEM images).
    """

    def __init__(
        self,
        root: PathLike,
        *,
        extensions: tuple[str, ...] = (".dm3", ".dm4", ".tif", ".tiff", ".ser"),
        recursive: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        normalize: bool = True,
        return_metadata: bool = True,
        frame_index: int | None = None,
    ) -> None:
        super().__init__(
            root,
            extensions=extensions,
            recursive=recursive,
            transform=transform,
            normalize=normalize,
            return_metadata=return_metadata,
            frame_index=frame_index,
        )


class FIBDataset(ScientificImageDataset):
    """Scientific image dataset specialized for FIB / SEM data.

    Defaults to formats commonly produced by FIB and SEM workflows
    (TIFF and PNG snapshots, plus DM-format slices when present).
    """

    def __init__(
        self,
        root: PathLike,
        *,
        extensions: tuple[str, ...] = (".tif", ".tiff", ".png", ".dm3", ".dm4"),
        recursive: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        normalize: bool = True,
        return_metadata: bool = True,
        frame_index: int | None = None,
    ) -> None:
        super().__init__(
            root,
            extensions=extensions,
            recursive=recursive,
            transform=transform,
            normalize=normalize,
            return_metadata=return_metadata,
            frame_index=frame_index,
        )


class UnlabeledScientificImageDataset(ScientificImageDataset):
    """Batched unlabeled image dataset for self-supervised pretraining.

    It filters out segmentation label files by default, converts all images to
    a fixed channel count, and optionally resizes them so the default
    ``DataLoader`` collate function works on mixed microscopy exports.
    """

    def __init__(
        self,
        root: PathLike,
        *,
        extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS,
        recursive: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        normalize: bool = True,
        return_metadata: bool = False,
        frame_index: int | None = None,
        channels: int = 1,
        image_size: int | tuple[int, int] | None = None,
        exclude_label_suffixes: tuple[str, ...] = (
            "_label.png",
            "_label.tif",
            "_label.tiff",
        ),
    ) -> None:
        super().__init__(
            root,
            extensions=extensions,
            recursive=recursive,
            transform=None,
            normalize=normalize,
            return_metadata=return_metadata,
            frame_index=frame_index,
        )
        self.paths = [
            p
            for p in self.paths
            if not any(
                p.name.lower().endswith(suffix)
                for suffix in exclude_label_suffixes
            )
        ]
        self.post_transform = transform
        self.channels = channels
        self.image_size = image_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        tensor = ensure_channel_count(sample["image"], self.channels)
        if self.image_size is not None:
            tensor = resize_chw(tensor, self.image_size, mode="bilinear")
        if self.post_transform is not None:
            tensor = self.post_transform(tensor)
        sample["image"] = tensor
        return sample


class SegmentationPairDataset(Dataset[dict[str, Any]]):
    """Image/mask pairs for fine-tuning and weakly supervised segmentation.

    Files named ``*_label.png`` are paired with the same stem without the
    ``_label`` suffix. Label values are remapped with one global mapping across
    the dataset, so class ids remain stable between samples.
    """

    def __init__(
        self,
        root: PathLike,
        *,
        image_size: int | tuple[int, int] | None = None,
        channels: int = 1,
        normalize: bool = True,
        recursive: bool = False,
        label_suffix: str = "_label.png",
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.channels = channels
        self.normalize = normalize
        globber = self.root.rglob if recursive else self.root.glob
        label_paths = sorted(globber(f"*{label_suffix}"))
        self.pairs: list[tuple[Path, Path]] = []
        for label_path in label_paths:
            image_name = label_path.name.removesuffix(label_suffix) + ".png"
            image_path = label_path.with_name(image_name)
            if image_path.exists():
                self.pairs.append((image_path.resolve(), label_path.resolve()))
        if not self.pairs:
            raise FileNotFoundError(f"No image/{label_suffix} pairs found under {root}")

        values: set[int] = set()
        for _, label_path in self.pairs:
            label, _ = load_image_array(label_path)
            if label.ndim == 3:
                label = label[..., 0]
            values.update(int(v) for v in np.unique(label))
        self.label_values = tuple(sorted(values))
        self.label_to_index = {value: idx for idx, value in enumerate(self.label_values)}

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def num_classes(self) -> int:
        return len(self.label_values)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, label_path = self.pairs[index]
        image, _ = load_image_array(image_path)
        label, _ = load_image_array(label_path)
        if self.normalize:
            image = _normalize_to_float(image)
        image_tensor = ensure_channel_count(_to_chw_tensor(image, None), self.channels)
        if label.ndim == 3:
            label = label[..., 0]
        mask = torch.empty(label.shape, dtype=torch.long)
        for old_value, new_index in self.label_to_index.items():
            mask[torch.from_numpy(label == old_value)] = new_index

        if self.image_size is not None:
            image_tensor = resize_chw(image_tensor, self.image_size, mode="bilinear")
            mask = (
                resize_chw(
                    mask.unsqueeze(0).float(),
                    self.image_size,
                    mode="nearest",
                )
                .squeeze(0)
                .long()
            )

        return {
            "image": image_tensor,
            "mask": mask,
            "path": str(image_path),
            "label_path": str(label_path),
        }


__all__ = [
    "FIBDataset",
    "ImageMetadata",
    "SegmentationPairDataset",
    "STEMDataset",
    "SUPPORTED_EXTENSIONS",
    "ScientificImageDataset",
    "UnlabeledScientificImageDataset",
    "ensure_channel_count",
    "load_image_array",
    "resize_chw",
]
