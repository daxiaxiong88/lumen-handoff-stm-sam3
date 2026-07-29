"""Safe executable entrypoint for ``stm_multilabel_experiment``.

The core experiment module returns NumPy scalar values from connected-component
evaluation. This wrapper supplies a JSON encoder for those scalar values before
delegating to its CLI, keeping the core experiment source untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import stm_multilabel_experiment as experiment

_ORIGINAL_DUMPS = experiment.json.dumps


def _json_default(value: object) -> object:
    """Convert only values that standard JSON cannot represent natively."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _dumps(payload: object, *args: Any, **kwargs: Any) -> str:
    """Add the experiment-safe encoder while preserving normal JSON options."""
    kwargs.setdefault("default", _json_default)
    return _ORIGINAL_DUMPS(payload, *args, **kwargs)


experiment.json.dumps = _dumps


if __name__ == "__main__":
    experiment.main()
