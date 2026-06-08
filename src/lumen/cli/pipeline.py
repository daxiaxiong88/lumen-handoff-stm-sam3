"""Declarative pipeline YAML support for the Lumen CLI."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from lumen.utils.config import _coerce


class PipelineModel(BaseModel):
    """Minimal Roboflow-inference-parity pipeline schema."""

    model_config = ConfigDict(extra="allow")

    source: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)
    filter: dict[str, Any] = Field(default_factory=dict)
    sample: dict[str, Any] = Field(default_factory=dict)
    sink: dict[str, Any] = Field(default_factory=dict)


class PipelineDocument(BaseModel):
    """Top-level pipeline YAML document."""

    model_config = ConfigDict(extra="allow")

    pipeline: PipelineModel


def load_pipeline_document(path: str | Path) -> PipelineDocument:
    """Load a pipeline YAML file and apply LUMEN__PIPELINE__* overrides."""
    pipeline_path = Path(path)
    with pipeline_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError("Pipeline YAML root must be a mapping")
    resolved = _apply_pipeline_env_overrides(raw)
    return PipelineDocument.model_validate(resolved)


def pipeline_plan(path: str | Path) -> dict[str, Any]:
    """Return the resolved, JSON-serializable dry-run plan."""
    doc = load_pipeline_document(path)
    return {
        "pipeline_path": str(Path(path)),
        "pipeline": doc.model_dump(mode="json"),
    }


def _apply_pipeline_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(raw)
    prefix = "LUMEN__PIPELINE__"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = ["pipeline", *[part.lower() for part in key[len(prefix) :].split("__")]]
        target: dict[str, Any] = resolved
        for part in parts[:-1]:
            current = target.get(part)
            if not isinstance(current, dict):
                current = {}
                target[part] = current
            target = current
        target[parts[-1]] = _coerce(value)
    return resolved


__all__ = ["PipelineDocument", "PipelineModel", "load_pipeline_document", "pipeline_plan"]
