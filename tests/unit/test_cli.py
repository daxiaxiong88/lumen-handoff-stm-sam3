"""Tests for the Lumen Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from lumen.cli.main import app

runner = CliRunner()


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
    assert plan["pipeline"]["pipeline"]["source"]["dataset"] == "override"
    assert plan["pipeline"]["pipeline"]["model"]["encoder"] == "eupe-pretrained"


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
