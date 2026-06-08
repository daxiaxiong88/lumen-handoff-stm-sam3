"""Thin compatibility wrapper around ``lumen predict``.

The benchmark-era example surface is now represented by the CLI. This wrapper
keeps the old example path executable while delegating directly to the real
command implementation.
"""

from __future__ import annotations

from lumen.cli.main import app

if __name__ == "__main__":
    app(prog_name="lumen")
