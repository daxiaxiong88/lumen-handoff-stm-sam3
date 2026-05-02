"""Name-based encoder factory.

Lets ``configs/default.yaml`` select a backbone by name without
importing the encoder class:

    from lumen.models import build_encoder
    encoder = build_encoder("eupe", embed_dim=384, depth=12)

Encoders register themselves at import time via the
:func:`register_encoder` decorator, so a registry entry exists for every
encoder that has been imported. See :mod:`lumen.models.eupe` and
:mod:`lumen.models.dinov3` for the canonical examples.
"""

from __future__ import annotations

from typing import Callable

from lumen.models.encoder_base import EncoderProtocol

EncoderFactory = Callable[..., EncoderProtocol]

_REGISTRY: dict[str, EncoderFactory] = {}


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
        _REGISTRY[name] = factory
        return factory

    return deco


def build_encoder(name: str, **kwargs: object) -> EncoderProtocol:
    """Instantiate the registered encoder ``name`` with ``kwargs``."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown encoder {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def list_encoders() -> list[str]:
    """Return the names of every registered encoder, sorted."""
    return sorted(_REGISTRY)


__all__ = ["build_encoder", "list_encoders", "register_encoder"]
