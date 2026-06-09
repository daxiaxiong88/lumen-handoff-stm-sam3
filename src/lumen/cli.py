"""Lumen CLI entry point.

Provides a unified interface for model management, serving, and training.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the lumen CLI."""
    parser = argparse.ArgumentParser(prog="lumen", description="Lumen: Scientific Image SSL Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the inference server")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--alias", help="Initial model alias to load")
    serve_parser.add_argument("--ckpt", help="Direct path to checkpoint to load (fallback for HYP-217)")

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn
        from lumen.serving.registry import registry
        
        if args.ckpt and args.alias:
            registry.register_alias(args.alias, args.ckpt)
        elif args.ckpt:
            registry.register_alias("default", args.ckpt)
            
        uvicorn.run("lumen.serving.app:app", host=args.host, port=args.port, reload=False)
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    print(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
