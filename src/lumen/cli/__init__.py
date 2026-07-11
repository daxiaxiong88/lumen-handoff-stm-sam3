"""Command line interface for Lumen."""

from __future__ import annotations

from collections.abc import Sequence

from lumen.cli.app import app

__all__ = ["app", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point for the Typer CLI.

    Runs ``app`` in non-standalone mode so we can translate Click's control
    flow into a plain integer exit code (what ``console_scripts`` and tests
    expect). Returns 0 on success.
    """
    import click

    try:
        app(args=list(argv) if argv is not None else None, standalone_mode=False)
    except SystemExit as exc:  # e.g. --help calls ctx.exit()
        code = exc.code
        if code is None or code is True:
            return 0
        return code if isinstance(code, int) else 1
    except click.exceptions.Abort:
        return 1
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    return 0
