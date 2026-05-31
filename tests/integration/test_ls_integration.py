"""Integration test for Label Studio client against a live server.

Skipped unless LS_URL and LS_API_KEY environment variables are set.
Run with:  LS_URL=http://localhost:8080 LS_API_KEY=xxx pytest tests/integration/
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from lumen.annotation.label_studio import LabelStudioConfig
from lumen.annotation.ls_client import LabelStudioClient
from lumen.annotation.store import LabellingTaskStore

LS_URL = os.environ.get("LS_URL", "")
LS_API_KEY = os.environ.get("LS_API_KEY", "")

skip_no_server = pytest.mark.skipif(
    not LS_URL or not LS_API_KEY,
    reason="LS_URL and LS_API_KEY env vars required for integration test",
)


@skip_no_server
class TestLiveIntegration:
    """Full round-trip against a self-hosted Label Studio instance."""

    def test_create_push_pull_review(self, tmp_path: Path) -> None:
        client = LabelStudioClient(LS_URL, LS_API_KEY)
        store = LabellingTaskStore(f"sqlite:///{tmp_path / 'integration.db'}")

        # 1. Bootstrap project
        project_id = client.bootstrap_project(
            "lumen-integration-test",
            "detection",
            ("particle", "defect"),
        )
        assert isinstance(project_id, int)

        # 2. Create test images
        image_paths: list[Path] = []
        for i in range(5):
            img = tmp_path / f"img_{i}.png"
            arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(img)
            image_paths.append(img)

        # 3. Push tasks with predictions for first 3
        config = LabelStudioConfig(
            task_type="detection",
            class_names=("particle", "defect"),
        )
        predictions: dict[str, list] = {}
        for p in image_paths[:3]:
            predictions[str(p)] = [
                MagicMock(class_name="particle", confidence=0.85, xyxy=(10, 5, 50, 40))
            ]

        client.push_tasks(
            project_id, image_paths, predictions, config=config,
        )

        # 4. Pull annotations
        annotations = client.pull_annotations(project_id)
        assert len(annotations) >= 5

        # 5. Simulate review: accept first, reject second
        task_ids = [a["id"] for a in annotations[:2]]
        client.set_status(task_ids[0], "accepted")
        client.set_status(task_ids[1], "rejected")

        # 6. Pull and verify Lumen-owned statuses in task metadata
        reviewed = client.pull_annotations(project_id)
        by_id = {r["id"]: r for r in reviewed}
        assert by_id[task_ids[0]].get("meta", {}).get("lumen_status") == "accepted"
        assert by_id[task_ids[1]].get("meta", {}).get("lumen_status") == "rejected"

        # 7. Store round-trip
        for ann in annotations[:3]:
            store.add_task(
                image_path=str(tmp_path / f"img_{annotations.index(ann)}.png"),
                project_id=project_id,
                ls_task_id=ann["id"],
                status="predicted",
            )
        stored = store.list_by_project(project_id)
        assert len(stored) == 3

        store.update_status(task_ids[0], "accepted")
        accepted = store.list_by_status("accepted")
        assert len(accepted) >= 1
