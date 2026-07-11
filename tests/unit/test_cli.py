"""Tests for the Lumen Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from lumen.cli.app import app

runner = CliRunner()


def test_console_script_main_is_callable_and_returns_exit_code() -> None:
    # Regression guard: the ``lumen`` console script resolves ``lumen.cli:main``,
    # which must be a callable returning an int (not the shadowed submodule).
    from importlib.metadata import entry_points

    from lumen.cli import main

    assert callable(main)
    assert main(["--help"]) == 0
    assert main(["model", "list"]) == 0

    scripts = entry_points(group="console_scripts")
    lumen_ep = next(ep for ep in scripts if ep.name == "lumen")
    assert lumen_ep.load() is main


def test_root_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "predict" in result.output
    assert "prelabel" in result.output
    assert "retrain" in result.output
    assert "sync-labels" in result.output
    assert "serve" in result.output
    assert "model" in result.output


def test_predict_help_is_informative() -> None:
    result = runner.invoke(app, ["predict", "--help"])

    assert result.exit_code == 0
    assert "Image path or glob pattern" in result.output
    assert "--config" in result.output
    assert "--output-dir" in result.output


def test_predict_through_zoo_model(tmp_path: Path) -> None:
    import numpy as np

    img = tmp_path / "cell.png"
    Image.fromarray(np.zeros((40, 48), dtype=np.uint8), mode="L").save(img)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        ["predict", str(img), "--model", "simple-segmentation", "--output-dir", str(out_dir)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model"] == "simple-segmentation"
    assert payload["task"] == "semantic_segmentation"
    assert (out_dir / "cell_mask.png").exists()


def test_model_list_includes_zoo_catalogue() -> None:
    result = runner.invoke(app, ["model", "list"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    zoo_ids = {m["model_id"] for m in payload["zoo"]}
    assert {"simple-segmentation", "sam3", "vision_banana"} <= zoo_ids


def test_prelabel_dry_run_resolves_pipeline_env(
    tmp_path: Path, monkeypatch
) -> None:
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        """
pipeline:
  source: {type: hyperdata, dataset: livecell, split: unlabeled}
  model: {encoder: eupe-pretrained, head: upernet, ckpt: weights/livecell/best.pt}
  filter: {confidence_gate: 0.7}
  sample: {active: {method: entropy, k: 200}}
  sink: {type: label_studio, project: LiveCELL}
"""
    )
    monkeypatch.setenv("LUMEN__PIPELINE__SOURCE__DATASET", "override")

    result = runner.invoke(app, ["prelabel", "run", str(pipeline), "--dry-run"])

    assert result.exit_code == 0
    plan = json.loads(result.output)
    assert plan["pipeline"]["source"]["dataset"] == "override"
    assert plan["pipeline"]["model"]["encoder"] == "eupe-pretrained"


def test_model_registry_add_list_promote(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LUMEN_MODEL_REGISTRY", str(tmp_path / "models.json"))

    add_result = runner.invoke(
        app,
        ["model", "registry", "add", "tiny", "--encoder", "eupe", "--head", "upernet"],
    )
    promote_result = runner.invoke(app, ["model", "promote", "tiny"])
    list_result = runner.invoke(app, ["model", "list"])

    assert add_result.exit_code == 0
    assert promote_result.exit_code == 0
    assert list_result.exit_code == 0
    payload = json.loads(list_result.output)
    assert payload["local"]["active"] == "tiny"
    assert payload["local"]["aliases"]["tiny"]["encoder"] == "eupe"


def test_predict_writes_mask_for_synthetic_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("L", (32, 32), color=128).save(image_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  patch_size: 16
  in_channels: 1
  embed_dim: 32
  depth: 1
  num_heads: 4
data:
  image_size: [32, 32]
downstream:
  segmentation:
    decoder: segmentation
    num_classes: 2
"""
    )
    output_dir = tmp_path / "predictions"

    result = runner.invoke(
        app,
        [
            "predict",
            str(image_path),
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert Path(payload["outputs"][0]).exists()
    assert Path(payload["outputs"][0]).name == "sample_mask.png"
