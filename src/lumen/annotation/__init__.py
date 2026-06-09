"""Human annotation and correction integrations."""

from __future__ import annotations

from lumen.annotation.label_studio import (
    LabelStudioConfig,
    build_label_config,
    export_corrected_labels,
    prediction_to_label_studio_result,
    write_label_studio_tasks,
)
from lumen.annotation.prelabel import (
    PrelabelPipelineConfig,
    PrelabelReport,
    PrelabelRunner,
)

__all__ = [
    "LabelStudioClient",
    "LabelStudioConfig",
    "LabellingTaskStore",
    "PrelabelPipelineConfig",
    "PrelabelReport",
    "PrelabelRunner",
    "build_label_config",
    "export_corrected_labels",
    "prediction_to_label_studio_result",
    "write_label_studio_tasks",
]


def __getattr__(name: str) -> object:
    """Lazy-import optional-dependency modules."""
    if name == "LabelStudioClient":
        from lumen.annotation.ls_client import LabelStudioClient
        return LabelStudioClient
    if name == "LabellingTaskStore":
        from lumen.annotation.store import LabellingTaskStore
        return LabellingTaskStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
