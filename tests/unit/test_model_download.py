from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from lumen.models import download as model_download
from lumen.utils.config import load_config


def test_eupe_legacy_weights_are_discovered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path
    legacy = repo / "weights" / "EUPE-ViT-S.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"checkpoint")
    monkeypatch.setattr(model_download, "repo_root", lambda: repo)

    path = model_download.ensure_model_asset("eupe", variant="vit_s")

    assert path == legacy


def test_missing_model_requires_explicit_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(model_download, "repo_root", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="download=True"):
        model_download.ensure_model_asset("dinov3")


def test_modelscope_snapshot_download_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_snapshot_download(*, model_id: str, local_dir: str) -> str:
        calls.append((model_id, local_dir))
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return local_dir

    modelscope = types.ModuleType("modelscope")
    modelscope.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    file_download = types.ModuleType("modelscope.hub.file_download")
    file_download.model_file_download = lambda **_: "unused"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modelscope", modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.hub.file_download", file_download)

    path = model_download.ensure_model_asset(
        "sam3",
        model_id="facebook/sam3",
        local_dir=tmp_path / "sam3",
        download=True,
    )

    assert path == tmp_path / "sam3"
    assert calls == [("facebook/sam3", str(tmp_path / "sam3"))]


def test_strict_unknown_config_rejects_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("training:\n  lr: 0.001\n  typo_key: true\n")

    with pytest.raises(ValueError, match="training.typo_key"):
        load_config(str(path), strict_unknown=True)

    cfg = load_config(str(path))
    assert cfg.training.lr == pytest.approx(0.001)
