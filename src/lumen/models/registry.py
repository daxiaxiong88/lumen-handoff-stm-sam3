"""Name-based factories for the Lumen model zoo.

Two parallel registries are exposed:

* **Encoders** — backbones that return ``(B, N, D)`` patch tokens and
  satisfy :class:`EncoderProtocol`. Used by every Lumen trainer.
* **Segmenters** — promptable mask-producing models that satisfy
  :class:`SegmenterProtocol`. Used as zero-shot tools and (later) for
  fine-tuning via :class:`SegmenterTrainer`.

Both surfaces share the same ``register_X`` / ``build_X`` /
``list_Xs`` shape so a config can flip backbones AND segmenters by
name with no import-time coupling:

    from lumen.models import build_encoder, build_segmenter
    encoder   = build_encoder("eupe", embed_dim=384, depth=12)
    segmenter = build_segmenter("sam3")

Encoders register themselves at import time via the
:func:`register_encoder` decorator (see :mod:`lumen.models.eupe` and
:mod:`lumen.models.dinov3`); segmenters do the same via
:func:`register_segmenter` (see :mod:`lumen.models.sam3`).
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from lumen.models.encoder_base import EncoderProtocol
from lumen.models.segmenter_base import SegmenterProtocol

EncoderFactory = Callable[..., EncoderProtocol]
SegmenterFactory = Callable[..., SegmenterProtocol]
HeadFactory = Callable[..., nn.Module]

_ENCODERS: dict[str, EncoderFactory] = {}
_SEGMENTERS: dict[str, SegmenterFactory] = {}
_HEADS: dict[str, HeadFactory] = {}


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def register_encoder(name: str) -> Callable[[EncoderFactory], EncoderFactory]:
    """Register an encoder factory under ``name``.

    Use as a decorator on a callable that returns an
    :class:`EncoderProtocol`-compatible instance:

        @register_encoder("eupe")
        def _make_eupe(**kwargs) -> EUPEEncoder:
            return EUPEEncoder(**kwargs)

    Re-registering the same name overwrites the previous entry, which
    is convenient for tests that want to swap a real encoder for a
    fake.
    """

    def deco(factory: EncoderFactory) -> EncoderFactory:
        _ENCODERS[name] = factory
        return factory

    return deco


def build_encoder(name: str, **kwargs: object) -> EncoderProtocol:
    """Instantiate the registered encoder ``name`` with ``kwargs``."""
    if name not in _ENCODERS:
        raise KeyError(f"Unknown encoder {name!r}; available: {sorted(_ENCODERS)}")
    return _ENCODERS[name](**kwargs)


def list_encoders() -> list[str]:
    """Return the names of every registered encoder, sorted."""
    return sorted(_ENCODERS)


# ---------------------------------------------------------------------------
# Segmenters
# ---------------------------------------------------------------------------


def register_segmenter(
    name: str,
) -> Callable[[SegmenterFactory], SegmenterFactory]:
    """Register a segmenter factory under ``name``.

    Same shape as :func:`register_encoder` — wrap a callable that
    returns a :class:`SegmenterProtocol`-compatible instance.
    """

    def deco(factory: SegmenterFactory) -> SegmenterFactory:
        _SEGMENTERS[name] = factory
        return factory

    return deco


def build_segmenter(name: str, **kwargs: object) -> SegmenterProtocol:
    """Instantiate the registered segmenter ``name`` with ``kwargs``."""
    if name not in _SEGMENTERS:
        raise KeyError(f"Unknown segmenter {name!r}; available: {sorted(_SEGMENTERS)}")
    return _SEGMENTERS[name](**kwargs)


def list_segmenters() -> list[str]:
    """Return the names of every registered segmenter, sorted."""
    return sorted(_SEGMENTERS)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


def register_head(name: str) -> Callable[[HeadFactory], HeadFactory]:
    """Register a downstream head factory under ``name``."""

    def deco(factory: HeadFactory) -> HeadFactory:
        _HEADS[name] = factory
        return factory

    return deco


def build_head(name: str, **kwargs: object) -> nn.Module:
    """Instantiate the registered head ``name`` with ``kwargs``."""
    if name not in _HEADS:
        raise KeyError(f"Unknown head {name!r}; available: {sorted(_HEADS)}")
    return _HEADS[name](**kwargs)


def list_heads() -> list[str]:
    """Return the names of every registered head, sorted."""
    return sorted(_HEADS)


__all__ = [
    "build_encoder",
    "build_head",
    "build_segmenter",
    "list_encoders",
    "list_heads",
    "list_segmenters",
    "register_encoder",
    "register_head",
    "register_segmenter",
]
