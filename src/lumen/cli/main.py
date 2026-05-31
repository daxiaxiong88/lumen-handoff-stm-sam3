"""Typer-powered command line interface for Lumen."""

from __future__ import annotations

import glob
import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import typer
from PIL import Image

import lumen.models  # noqa: F401 - populate model registries on import
from lumen.cli.model_registry import (
    add_alias,
    load_registry,
    promote_alias,
    registry_path,
)
from lumen.cli.pipeline import pipeline_plan
from lumen.data.dataset import load_image_array
from lumen.inference import InferenceConfig, MicroscopyInference
from lumen.models import list_encoders, list_heads
from lumen.models.download import main as download_models_main
from lumen.utils.config import LumenConfig, load_config, load_config_from_env


class DeviceChoice(str, Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class TaskChoice(str, Enum):
    classification = "classification"
    segmentation = "segmentation"
    detection = "detection"


class OutputFormat(str, Enum):
    png = "png"
    json = "json"


DeviceOption = Annotated[
    DeviceChoice | None,
    typer.Option("--device", help="Execution device override."),
]
ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", exists=True, readable=True, help="Lumen YAML config."),
]
OutputDirOption = Annotated[
    Path,
    typer.Option("--output-dir", help="Directory for CLI outputs."),
]

app = typer.Typer(help="Lumen microscopy inference and pipeline runner.")
prelabel_app = typer.Typer(help="Pre-labeling pipeline commands.")
retrain_app = typer.Typer(help="Human-review retraining pipeline commands.")
sync_labels_app = typer.Typer(help="Label Studio synchronization commands.")
model_app = typer.Typer(help="Local model registry commands.")
model_registry_app = typer.Typer(help="Manage local model aliases.")

app.add_typer(prelabel_app, name="prelabel")
app.add_typer(retrain_app, name="retrain")
app.add_typer(sync_labels_app, name="sync-labels")
app.add_typer(model_app, name="model")
model_app.add_typer(model_registry_app, name="registry")


@app.command()
def predict(
    image_or_glob: Annotated[str, typer.Argument(help="Image path or glob pattern.")],
    config: ConfigOption = None,
    device: DeviceOption = None,
    output_dir: OutputDirOption = Path("outputs/predictions"),
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", exists=True, readable=True, help="Model checkpoint."),
    ] = None,
    encoder: Annotated[str | None, typer.Option("--encoder", help="Registered encoder name.")] = None,
    head: Annotated[str | None, typer.Option("--head", help="Registered head name.")] = None,
    task: Annotated[
        TaskChoice | None,
        typer.Option("--task", help="Prediction task type."),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Prediction output format."),
    ] = OutputFormat.png,
) -> None:
    """Run microscopy inference once for one image or a glob of images."""
    cfg = _load_lumen_config(config)
    paths = _resolve_image_paths(image_or_glob)
    if not paths:
        raise typer.BadParameter(f"No images matched: {image_or_glob}")
    output_dir.mkdir(parents=True, exist_ok=True)

    inference = MicroscopyInference(
        InferenceConfig(
            checkpoint_path=checkpoint,
            encoder_name=encoder or cfg.get("pipeline.model.encoder") or "simple",
            head_name=head or cfg.downstream.segmentation.decoder,
            task_type=(task.value if task is not None else "segmentation"),
            device=(device.value if device is not None else _resolve_device(cfg.device)),
            image_size=tuple(cfg.data.image_size),
            encoder_kwargs=_encoder_kwargs(cfg),
            num_classes=cfg.downstream.segmentation.num_classes,
        )
    )
    written: list[str] = []
    for path in paths:
        arr, _ = load_image_array(path)
        result = inference.infer(arr)
        written.append(str(_write_prediction(path, result.predictions, output_dir, output_format.value)))
    typer.echo(json.dumps({"images": [str(path) for path in paths], "outputs": written}, indent=2))


@prelabel_app.command("run")
def prelabel_run(
    pipeline_yaml: Annotated[Path, typer.Argument(exists=True, readable=True)],
    config: ConfigOption = None,
    device: DeviceOption = None,
    output_dir: OutputDirOption = Path("outputs/prelabel"),
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate and print the resolved plan.")] = False,
) -> None:
    """Run or dry-run a pre-labeling pipeline."""
    _ = (config, device, output_dir)
    plan = pipeline_plan(pipeline_yaml)
    if dry_run:
        typer.echo(json.dumps(plan, indent=2, sort_keys=True))
        return
    try:
        from lumen.prelabel import PrelabelRunner  # type: ignore[import-not-found]
    except ImportError as exc:
        raise typer.ClickException(
            "PrelabelRunner is not available yet. Re-run with --dry-run to validate YAML."
        ) from exc
    PrelabelRunner.from_plan(plan).run()  # type: ignore[name-defined]


@retrain_app.command("run")
def retrain_run(
    pipeline_yaml: Annotated[Path, typer.Argument(exists=True, readable=True)],
    config: ConfigOption = None,
    device: DeviceOption = None,
    output_dir: OutputDirOption = Path("outputs/retrain"),
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate and print the resolved plan.")] = False,
) -> None:
    """Run or dry-run a review-loop retraining pipeline."""
    _ = (config, device, output_dir)
    plan = pipeline_plan(pipeline_yaml)
    if dry_run:
        typer.echo(json.dumps(plan, indent=2, sort_keys=True))
        return
    try:
        from lumen.review_loop import ReviewLoop  # type: ignore[import-not-found]
    except ImportError as exc:
        raise typer.ClickException(
            "ReviewLoop is not available yet. Re-run with --dry-run to validate YAML."
        ) from exc
    ReviewLoop.from_plan(plan).run()  # type: ignore[name-defined]


@sync_labels_app.command("pull")
def sync_labels_pull(
    project: Annotated[str, typer.Option("--project", help="Label Studio project id.")],
    config: ConfigOption = None,
    device: DeviceOption = None,
    output_dir: OutputDirOption = Path("outputs/labels"),
) -> None:
    """Pull corrected labels from Label Studio when the client is available."""
    _ = (project, config, device, output_dir)
    raise typer.ClickException(
        "Label Studio API sync is not available in this checkout. Use exported JSON with lumen.annotation helpers."
    )


@sync_labels_app.command("push")
def sync_labels_push(
    project: Annotated[str, typer.Option("--project", help="Label Studio project id.")],
    config: ConfigOption = None,
    device: DeviceOption = None,
    output_dir: OutputDirOption = Path("outputs/labels"),
) -> None:
    """Push preannotations to Label Studio when the client is available."""
    _ = (project, config, device, output_dir)
    raise typer.ClickException(
        "Label Studio API sync is not available in this checkout. Use lumen.annotation helpers to write task JSON."
    )


@app.command()
def serve(
    config: ConfigOption = None,
    device: DeviceOption = None,
    output_dir: OutputDirOption = Path("outputs/server"),
) -> None:
    """Start the inference server when the server package is available."""
    _ = (config, device, output_dir)
    raise typer.ClickException(
        "lumen serve depends on the HTTP server work from HYP-218, which is not available in this checkout."
    )


@model_app.command("list")
def model_list() -> None:
    """List registered encoders, heads, and local aliases."""
    typer.echo(
        json.dumps(
            {
                "encoders": list_encoders(),
                "heads": list_heads(),
                "registry_path": str(registry_path()),
                "local": load_registry(),
            },
            indent=2,
            sort_keys=True,
        )
    )


@model_registry_app.command("add")
def model_registry_add(
    name: Annotated[str, typer.Argument(help="Alias name.")],
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", help="Checkpoint path.")] = None,
    encoder: Annotated[str, typer.Option("--encoder", help="Registered encoder name.")] = "eupe",
    head: Annotated[str, typer.Option("--head", help="Registered head name.")] = "upernet",
    task: Annotated[str, typer.Option("--task", help="Task type.")] = "segmentation",
) -> None:
    """Add or replace a local model alias."""
    entry = add_alias(
        name,
        checkpoint=str(checkpoint) if checkpoint is not None else None,
        encoder=encoder,
        head=head,
        task=task,
    )
    typer.echo(json.dumps({"name": name, "entry": entry}, indent=2, sort_keys=True))


@model_app.command("promote")
def model_promote(name: Annotated[str, typer.Argument(help="Alias to mark active.")]) -> None:
    """Promote a local model alias to active."""
    try:
        promote_alias(name)
    except KeyError as exc:
        raise typer.ClickException(str(exc)) from exc
    typer.echo(json.dumps({"active": name}, indent=2))


@app.command("download-models")
def download_models() -> None:
    """Download configured model assets."""
    download_models_main()


def _load_lumen_config(config: Path | None) -> LumenConfig:
    return load_config(str(config)) if config is not None else load_config_from_env()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cpu"
    return device


def _encoder_kwargs(cfg: LumenConfig) -> dict[str, object]:
    return {
        "patch_size": cfg.model.patch_size,
        "in_channels": cfg.model.in_channels,
        "embed_dim": cfg.model.embed_dim,
        "depth": cfg.model.depth,
        "num_heads": cfg.model.num_heads,
        "mlp_ratio": cfg.model.mlp_ratio,
        "dropout": cfg.model.dropout,
    }


def _resolve_image_paths(pattern: str) -> list[Path]:
    matches = glob.glob(pattern)
    if matches:
        return [Path(match) for match in sorted(matches)]
    path = Path(pattern)
    return [path] if path.exists() else []


def _write_prediction(
    image_path: Path,
    prediction: object,
    output_dir: Path,
    output_format: Literal["png", "json"],
) -> Path:
    stem = image_path.stem
    arr = np.asarray(prediction)
    if output_format == "json":
        out = output_dir / f"{stem}_prediction.json"
        out.write_text(json.dumps({"image": str(image_path), "prediction": arr.tolist()}, indent=2))
        return out
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    out = output_dir / f"{stem}_mask.png"
    Image.fromarray(arr.astype(np.uint8), mode="L").save(out)
    return out


if __name__ == "__main__":
    app()
