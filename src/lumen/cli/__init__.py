"""Command line interface for Lumen workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from lumen.retrain import ModelRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lumen")
    sub = parser.add_subparsers(dest="command", required=True)

    retrain = sub.add_parser("retrain")
    retrain_sub = retrain.add_subparsers(dest="subcommand", required=True)
    run = retrain_sub.add_parser("run")
    run.add_argument("pipeline")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--registry-root", default="weights/registry")

    model = sub.add_parser("model")
    model_sub = model.add_subparsers(dest="subcommand", required=True)
    promote = model_sub.add_parser("promote")
    promote.add_argument("ckpt")
    promote.add_argument("--as", dest="alias", required=True)
    promote.add_argument("--on", dest="gate", default="miou_delta>=+0.01")
    promote.add_argument("--metrics", default=None)
    promote.add_argument("--registry-root", default="weights/registry")
    list_cmd = model_sub.add_parser("list")
    list_cmd.add_argument("--registry-root", default="weights/registry")

    args = parser.parse_args(argv)
    if args.command == "retrain" and args.subcommand == "run":
        return _retrain_run(args)
    if args.command == "model" and args.subcommand == "promote":
        return _model_promote(args)
    if args.command == "model" and args.subcommand == "list":
        return _model_list(args)
    parser.error("unsupported command")
    return 2


def _retrain_run(args: argparse.Namespace) -> int:
    pipeline = _load_yaml(Path(args.pipeline))
    plan = _pipeline_plan(pipeline, ModelRegistry(args.registry_root))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    raise NotImplementedError("Non-dry-run retraining requires a project-specific trainer factory.")


def _model_promote(args: argparse.Namespace) -> int:
    metrics = _load_metrics(args.metrics, args.ckpt)
    registry = ModelRegistry(args.registry_root)
    promoted = registry.promote(args.ckpt, alias=args.alias, metrics=metrics, gate=args.gate)
    print(json.dumps({"promoted": promoted, "alias": args.alias}, indent=2))
    return 0 if promoted else 1


def _model_list(args: argparse.Namespace) -> int:
    registry = ModelRegistry(args.registry_root)
    print(json.dumps(registry.list(), indent=2))
    return 0


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Pipeline YAML must contain a mapping")
    return data


def _pipeline_plan(data: dict[str, Any], registry: ModelRegistry) -> dict[str, Any]:
    pipe = data.get("pipeline") or {}
    source = pipe.get("source") or {}
    trainer = pipe.get("trainer") or {}
    eval_cfg = pipe.get("eval") or {}
    promote = pipe.get("promote") or {}
    if isinstance(promote, dict) and True in promote and "on" not in promote:
        promote = dict(promote)
        promote["on"] = promote.pop(True)
    base = str(trainer.get("base_ckpt", ""))
    return {
        "source": source,
        "base_ckpt": base,
        "resolved_base_ckpt": registry.resolve(base) if base else "",
        "trainer": trainer,
        "eval": eval_cfg,
        "promote": promote,
    }


def _load_metrics(path: str | None, ckpt: str | None = None) -> dict[str, float]:
    if path is not None:
        data = json.loads(Path(path).read_text())
        return {str(k): float(v) for k, v in data.items()}
    if ckpt is None or not Path(ckpt).exists():
        return {}
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("metrics"), dict):
        return {str(k): float(v) for k, v in state["metrics"].items()}
    return {}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
