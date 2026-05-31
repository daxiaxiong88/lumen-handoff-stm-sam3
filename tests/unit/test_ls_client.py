"""Unit tests for ls_client.py and store.py (HYP-214)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests_mock as rm_module  # type: ignore[import-untyped]
from PIL import Image

from lumen.annotation.label_studio import LabelStudioConfig, build_label_config
from lumen.annotation.ls_client import LabelStudioClient
from lumen.annotation.store import LabellingTaskStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LS_URL = "http://localhost:8080"
LS_KEY = "test-api-key-123"


@pytest.fixture
def client() -> LabelStudioClient:
    return LabelStudioClient(LS_URL, LS_KEY)


@pytest.fixture
def mock_ls():
    return rm_module.Mocker()


@pytest.fixture
def detection_config() -> LabelStudioConfig:
    return LabelStudioConfig(
        task_type="detection",
        class_names=("particle", "defect"),
    )


@pytest.fixture
def tmp_image(tmp_path: Path) -> Path:
    img = tmp_path / "test.png"
    arr = np.zeros((20, 30), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(img)
    return img


@pytest.fixture
def store(tmp_path: Path) -> LabellingTaskStore:
    db_path = tmp_path / "test.db"
    return LabellingTaskStore(f"sqlite:///{db_path}")


# ---------------------------------------------------------------------------
# LabelStudioClient — bootstrap_project
# ---------------------------------------------------------------------------

class TestBootstrapProject:
    def test_creates_new_project(self, client: LabelStudioClient, mock_ls) -> None:
        mock_ls.get(f"{LS_URL}/api/projects/", json=[])
        mock_ls.post(
            f"{LS_URL}/api/projects/",
            json={"id": 42, "title": "My Project"},
        )

        with mock_ls:
            pid = client.bootstrap_project("My Project", "detection", ("particle",))
        assert pid == 42

    def test_returns_existing_project(self, client: LabelStudioClient, mock_ls) -> None:
        config = LabelStudioConfig(task_type="classification", class_names=("cat", "dog"))
        label_config = build_label_config(config)
        mock_ls.get(
            f"{LS_URL}/api/projects/",
            json=[{"id": 7, "title": "Existing", "label_config": label_config}],
        )
        mock_ls.get(f"{LS_URL}/api/projects/7/", json={"label_config": label_config})

        with mock_ls:
            pid = client.bootstrap_project("Existing", "classification", ("cat", "dog"))
        assert pid == 7

    def test_updates_config_on_mismatch(self, client: LabelStudioClient, mock_ls) -> None:
        old_config = "<View>old</View>"
        mock_ls.get(
            f"{LS_URL}/api/projects/",
            json=[{"id": 7, "title": "Proj", "label_config": old_config}],
        )
        mock_ls.get(f"{LS_URL}/api/projects/7/", json={"label_config": old_config})
        mock_ls.patch(f"{LS_URL}/api/projects/7/", json={"id": 7})

        with mock_ls:
            pid = client.bootstrap_project("Proj", "detection", ("particle",))
        assert pid == 7


# ---------------------------------------------------------------------------
# LabelStudioClient — push_tasks
# ---------------------------------------------------------------------------

class TestPushTasks:
    def test_push_without_predictions(
        self, client: LabelStudioClient, mock_ls, tmp_image: Path,
    ) -> None:
        mock_ls.post(f"{LS_URL}/api/projects/1/import", json=[])

        with mock_ls:
            result = client.push_tasks(1, [str(tmp_image)])
        assert len(result) == 1
        assert result[0]["data"]["image"]

    def test_push_with_predictions(
        self,
        client: LabelStudioClient,
        mock_ls,
        tmp_image: Path,
        detection_config: LabelStudioConfig,
    ) -> None:
        mock_ls.post(f"{LS_URL}/api/projects/1/import", json=[])

        pred = MagicMock(class_name="particle", confidence=0.9, xyxy=(5, 2, 20, 10))

        with mock_ls:
            result = client.push_tasks(
                1, [str(tmp_image)],
                {str(tmp_image.resolve()): [pred]},
                config=detection_config,
            )
        assert len(result) == 1
        assert "predictions" in result[0]
        assert result[0]["predictions"][0]["result"]

    def test_push_empty_list(self, client: LabelStudioClient, mock_ls) -> None:
        with mock_ls:
            result = client.push_tasks(1, [])
        assert result == []

    def test_push_tiff_generates_preview(
        self, client: LabelStudioClient, mock_ls, tmp_path: Path,
    ) -> None:
        mock_ls.post(f"{LS_URL}/api/projects/1/import", json=[])

        tiff_path = tmp_path / "sample.tiff"
        arr = np.random.randint(0, 65535, (32, 32), dtype=np.uint16)
        Image.fromarray(arr).save(tiff_path)

        with mock_ls:
            result = client.push_tasks(1, [str(tiff_path)])
        assert len(result) == 1
        assert result[0]["data"]["image"].endswith(".preview.png")
        assert "original_path" in result[0]["meta"]
        preview_path = tiff_path.with_suffix(".preview.png")
        assert preview_path.exists()

    def test_push_png_no_preview(
        self, client: LabelStudioClient, mock_ls, tmp_image: Path,
    ) -> None:
        mock_ls.post(f"{LS_URL}/api/projects/1/import", json=[])

        with mock_ls:
            result = client.push_tasks(1, [str(tmp_image)])
        assert len(result) == 1
        assert result[0]["data"]["image"].endswith(".png")
        assert "original_path" not in result[0]["meta"]


# ---------------------------------------------------------------------------
# LabelStudioClient — pull_annotations
# ---------------------------------------------------------------------------

class TestPullAnnotations:
    def test_pull_all(self, client: LabelStudioClient, mock_ls) -> None:
        export = [
            {"id": 1, "annotations": [{"result": []}]},
            {"id": 2, "annotations": [{"result": []}]},
        ]
        mock_ls.get(f"{LS_URL}/api/projects/1/export", json=export)

        with mock_ls:
            result = client.pull_annotations(1)
        assert len(result) == 2

    def test_pull_with_since(self, client: LabelStudioClient, mock_ls) -> None:
        mock_ls.get(f"{LS_URL}/api/projects/1/export", json=[])

        with mock_ls:
            result = client.pull_annotations(1, since="2026-01-01T00:00:00Z")
        assert result == []


# ---------------------------------------------------------------------------
# LabelStudioClient — status management
# ---------------------------------------------------------------------------

class TestStatusManagement:
    def test_set_status(self, client: LabelStudioClient, mock_ls) -> None:
        mock_ls.patch(f"{LS_URL}/api/tasks/10/", json={"id": 10, "status": "accepted"})

        with mock_ls:
            result = client.set_status(10, "accepted")
        assert result["status"] == "accepted"

    def test_mark_reviewed(self, client: LabelStudioClient, mock_ls) -> None:
        mock_ls.patch(f"{LS_URL}/api/tasks/1/", json={"id": 1, "status": "accepted"})
        mock_ls.patch(f"{LS_URL}/api/tasks/2/", json={"id": 2, "status": "accepted"})

        with mock_ls:
            results = client.mark_reviewed([1, 2])
        assert len(results) == 2


# ---------------------------------------------------------------------------
# LabellingTaskStore — round-trip
# ---------------------------------------------------------------------------

class TestLabellingTaskStore:
    def test_add_and_get(self, store: LabellingTaskStore) -> None:
        task = store.add_task(
            image_path="/tmp/img.png",
            project_id=1,
            ls_task_id=42,
            status="predicted",
            model_version="lumen-v1",
        )
        assert task.image_path == "/tmp/img.png"
        assert task.project_id == 1
        assert task.ls_task_id == 42
        assert task.status == "predicted"

        fetched = store.get_by_ls_task_id(42)
        assert fetched is not None
        assert fetched.image_path == "/tmp/img.png"

    def test_update_status(self, store: LabellingTaskStore) -> None:
        store.add_task(image_path="/tmp/a.png", project_id=1, ls_task_id=10)
        updated = store.update_status(10, "accepted")
        assert updated is not None
        assert updated.status == "accepted"

        fetched = store.get_by_ls_task_id(10)
        assert fetched is not None
        assert fetched.status == "accepted"

    def test_update_status_rejects_invalid(self, store: LabellingTaskStore) -> None:
        with pytest.raises(ValueError, match="Invalid status"):
            store.update_status(1, "bogus")

    def test_get_by_image_path(self, store: LabellingTaskStore) -> None:
        store.add_task(image_path="/tmp/x.png", project_id=1)
        store.add_task(image_path="/tmp/x.png", project_id=2)
        store.add_task(image_path="/tmp/y.png", project_id=1)

        results = store.get_by_image_path("/tmp/x.png")
        assert len(results) == 2

    def test_list_by_status(self, store: LabellingTaskStore) -> None:
        store.add_task(image_path="/tmp/a.png", project_id=1, status="predicted")
        store.add_task(image_path="/tmp/b.png", project_id=1, status="unlabelled")
        store.add_task(image_path="/tmp/c.png", project_id=1, status="predicted")

        predicted = store.list_by_status("predicted")
        assert len(predicted) == 2

    def test_list_by_project(self, store: LabellingTaskStore) -> None:
        store.add_task(image_path="/tmp/a.png", project_id=1)
        store.add_task(image_path="/tmp/b.png", project_id=2)
        store.add_task(image_path="/tmp/c.png", project_id=1)

        results = store.list_by_project(1)
        assert len(results) == 2

    def test_set_ls_task_id(self, store: LabellingTaskStore) -> None:
        store.add_task(image_path="/tmp/z.png", project_id=5)
        updated = store.set_ls_task_id("/tmp/z.png", 5, 99)
        assert updated is not None
        assert updated.ls_task_id == 99

    def test_get_nonexistent(self, store: LabellingTaskStore) -> None:
        assert store.get_by_ls_task_id(9999) is None

    def test_add_task_rejects_invalid_status(self, store: LabellingTaskStore) -> None:
        with pytest.raises(ValueError, match="Invalid status"):
            store.add_task(image_path="/tmp/bad.png", project_id=1, status="invalid")
