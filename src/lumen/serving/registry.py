"""Thin model registry wrapper for serving.

Resolves model aliases to checkpoint paths. Since HYP-217 (model registry)
might not be fully merged, this provides a fallback mechanism.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry to resolve model aliases to checkpoint paths."""

    def __init__(self, default_ckpt_dir: str | Path | None = None) -> None:
        self.default_ckpt_dir = Path(default_ckpt_dir) if default_ckpt_dir else None
        self._aliases: dict[str, Path] = {}

    def register_alias(self, alias: str, checkpoint_path: str | Path) -> None:
        """Register a model alias."""
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"Checkpoint path does not exist for alias {alias!r}: {path}")
        self._aliases[alias] = path
        logger.info(f"Registered alias {alias!r} -> {path}")

    def resolve_alias(self, alias_or_path: str) -> Path:
        """Resolve an alias or return the path if it's already a valid path."""
        # Check if it's a registered alias
        if alias_or_path in self._aliases:
            return self._aliases[alias_or_path]

        # Check if it's a direct path
        path = Path(alias_or_path)
        if path.exists():
            return path

        # Try to resolve from default_ckpt_dir if provided
        if self.default_ckpt_dir:
            resolved = self.default_ckpt_dir / alias_or_path
            if resolved.exists():
                return resolved

        raise ValueError(f"Could not resolve model alias or path: {alias_or_path!r}")

    def list_aliases(self) -> list[str]:
        """List all registered aliases."""
        return sorted(self._aliases.keys())


# Global registry instance
registry = ModelRegistry()
