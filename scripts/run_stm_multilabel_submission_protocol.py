"""Protocol runner that dispatches the NumPy-safe STM experiment entrypoint."""

from __future__ import annotations

from pathlib import Path

import run_stm_multilabel_protocol as protocol

_ORIGINAL_COMMAND_FOR = protocol.command_for


def _command_for(
    job: protocol.Job, config: dict[str, object], experiment_script: Path
) -> list[str]:
    """Replace only the executable path, retaining the checked protocol options."""
    command = _ORIGINAL_COMMAND_FOR(job, config, experiment_script)
    command[1] = str(Path(__file__).with_name("stm_multilabel_submission.py"))
    return command


protocol.command_for = _command_for


if __name__ == "__main__":
    protocol.main()
