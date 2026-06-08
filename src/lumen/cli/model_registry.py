"""Local model alias registry used by the CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def registry_path() -> Path:
    override = os.environ.get("LUMEN_MODEL_REGISTRY")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "lumen" / "model_registry.json"


def load_registry() -> dict[str, Any]:
    path = registry_path()
    if not path.exists():
        return {"aliases": {}, "active": None}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Model registry must contain a JSON object: {path}")
    data.setdefault("aliases", {})
    data.setdefault("active", None)
    return data


def save_registry(data: dict[str, Any]) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def add_alias(
    name: str,
    *,
    checkpoint: str | None = None,
    encoder: str = "eupe",
    head: str = "upernet",
    task: str = "segmentation",
) -> dict[str, Any]:
    data = load_registry()
    aliases = data.setdefault("aliases", {})
    aliases[name] = {
        "checkpoint": checkpoint,
        "encoder": encoder,
        "head": head,
        "task": task,
    }
    save_registry(data)
    return aliases[name]


def promote_alias(name: str) -> None:
    data = load_registry()
    aliases = data.setdefault("aliases", {})
    if name not in aliases:
        raise KeyError(f"Unknown model alias: {name}")
    data["active"] = name
    save_registry(data)


__all__ = ["add_alias", "load_registry", "promote_alias", "registry_path", "save_registry"]
