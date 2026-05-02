"""Encoder contract used by all Lumen trainers and heads.

Lumen trainers used to be typed against :class:`EUPEEncoder` directly,
which made it impossible for a type checker to recognise that
:class:`DINOv3Encoder` (or any future custom backbone) was a valid
substitute. This module pins down the structural contract that every
encoder must satisfy:

* ``forward(x)`` returns patch tokens shaped ``(B, N, D)``.
* ``embed_dim``, ``patch_size``, ``in_channels`` are concrete ``int``
  attributes — heads and the MAE patchifier read them.

Two surfaces are provided:

* :class:`EncoderProtocol` — a :class:`typing.Protocol` for *structural*
  matching. Trainers and heads accept this, so any third-party encoder
  with the right shape is valid without inheritance.
* :class:`EncoderBase` — an :class:`torch.nn.Module` ABC for *in-house*
  encoders. Subclasses get the protocol attributes and have to implement
  ``forward``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class EncoderProtocol(Protocol):
    """Structural type every Lumen-compatible encoder must satisfy."""

    embed_dim: int
    patch_size: int
    in_channels: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class EncoderBase(nn.Module):
    """ABC for in-house Lumen encoders.

    Subclasses set ``embed_dim``, ``patch_size``, ``in_channels`` in
    ``__init__`` and implement ``forward(x) -> Tensor[B, N, D]``.
    """

    embed_dim: int
    patch_size: int
    in_channels: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        del x
        raise NotImplementedError


__all__ = ["EncoderBase", "EncoderProtocol"]
