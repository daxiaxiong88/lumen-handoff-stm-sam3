"""Live Label Studio API client for pushing tasks and pulling annotations."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from lumen.annotation.label_studio import (
    LabelStudioConfig,
    TaskType,
    build_label_config,
    prediction_to_label_studio_result,
)
from lumen.data.dataset import load_image_array
from lumen.data.supervision_bridge import prepare_image_for_supervision

_NON_DISPLAY_EXTENSIONS = frozenset({".tiff", ".tif", ".dm3", ".dm4"})

logger = logging.getLogger(__name__)


class LabelStudioClient:
    """Client for a self-hosted Label Studio instance.

    Wraps the Label Studio REST API to create projects, push tasks with
    model predictions, and pull human-corrected annotations.
    """

    def __init__(self, url: str, api_key: str) -> None:
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Token {api_key}"})
        self._project_configs: dict[int, LabelStudioConfig] = {}

    @property
    def base_url(self) -> str:
        return self._url

    def health(self) -> bool:
        """Check Label Studio server connectivity."""
        try:
            resp = self._session.get(f"{self._url}/api/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    def bootstrap_project(
        self,
        name: str,
        task_type: TaskType,
        class_names: tuple[str, ...] | list[str],
    ) -> int:
        """Idempotently create a Label Studio project.

        If a project with the same title already exists, returns its id.
        """
        config = LabelStudioConfig(
            task_type=task_type,
            class_names=tuple(class_names),
        )
        label_config = build_label_config(config)

        # Check for existing project with the same title. Label Studio returns
        # paginated responses on current versions, but older mocks/installations
        # may still return a bare list.
        next_url: str | None = f"{self._url}/api/projects/"
        params: dict[str, Any] | None = {"title": name}
        while next_url is not None:
            resp = self._session.get(next_url, params=params)
            resp.raise_for_status()
            body = resp.json()
            for proj in _items_from_response(body):
                if proj.get("title") == name:
                    project_id = int(proj["id"])
                    self._project_configs[project_id] = config
                    # Update labelling config if it differs
                    self._maybe_update_config(project_id, label_config)
                    return project_id
            next_url = _next_url(body, self._url)
            params = None

        resp = self._session.post(
            f"{self._url}/api/projects/",
            json={"title": name, "label_config": label_config},
        )
        resp.raise_for_status()
        project_id = int(resp.json()["id"])
        self._project_configs[project_id] = config
        return project_id

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def push_tasks(
        self,
        project_id: int,
        image_paths: list[str | Path],
        predictions_by_image: dict[str, list[Any]] | None = None,
        *,
        config: LabelStudioConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Upload tasks to a project, optionally with model predictions.

        For PNG/JPEG images the URL is passed by reference. For TIFF/DM3
        (formats Label Studio cannot render), a percentile-stretched PNG
        preview is generated and the original path is stored in task meta.
        """
        predictions_by_image = predictions_by_image or {}
        if config is None:
            config = self._project_configs.get(project_id)
        if predictions_by_image and config is None:
            raise ValueError(
                "config is required when predictions_by_image contains predictions; "
                "call bootstrap_project() first or pass config=..."
            )
        tasks: list[dict[str, Any]] = []

        for raw_path in image_paths:
            image_path = Path(raw_path).resolve()
            arr, _ = load_image_array(image_path)
            height, width = _image_size(arr)

            needs_preview = image_path.suffix.lower() in _NON_DISPLAY_EXTENSIONS
            if needs_preview:
                preview_path = image_path.with_suffix(".preview.png")
                preview_arr = prepare_image_for_supervision(arr)
                Image.fromarray(preview_arr).save(preview_path)
                image_url = str(preview_path)
            else:
                image_url = str(image_path)

            task_data: dict[str, Any] = {
                "data": {"image": image_url},
                "meta": {
                    "image_path": str(image_path),
                    "width": width,
                    "height": height,
                },
            }
            if needs_preview:
                task_data["meta"]["original_path"] = str(image_path)

            # Build predictions if provided
            preds = predictions_by_image.get(str(image_path), [])
            if preds:
                results: list[dict[str, Any]] = []
                for pred in preds:
                    result = prediction_to_label_studio_result(
                        pred, image_size=(height, width), config=config,
                    )
                    if result is not None:
                        results.append(result)
                if results:
                    pred_hash = hashlib.md5(
                        json.dumps(results, sort_keys=True).encode()
                    ).hexdigest()[:12]
                    task_data["predictions"] = [
                        {"model_version": "lumen", "result": results}
                    ]
                    task_data["meta"]["prediction_hash"] = pred_hash

            tasks.append(task_data)

        if not tasks:
            return []

        # Use the import endpoint
        resp = self._session.post(
            f"{self._url}/api/projects/{project_id}/import",
            json=tasks,
        )
        resp.raise_for_status()
        body = resp.json()
        # The import endpoint may return a list of task dicts or just ids
        return body if isinstance(body, list) and len(body) == len(tasks) else tasks

    # ------------------------------------------------------------------
    # Annotation export
    # ------------------------------------------------------------------

    def pull_annotations(
        self,
        project_id: int,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """Export annotations from a project.

        Args:
            project_id: Label Studio project ID.
            since: Optional ISO-8601 timestamp to filter by updated_at.

        Returns:
            List of task dicts from the Label Studio export endpoint.
        """
        resp = self._session.get(
            f"{self._url}/api/projects/{project_id}/export",
            params={"exportType": "JSON"},
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return []
        if since is None:
            return data

        cutoff = _parse_iso8601(since)
        return [
            task
            for task in data
            if (updated_at := _parse_optional_iso8601(task.get("updated_at"))) is not None
            and updated_at >= cutoff
        ]

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def set_status(self, task_id: int, status: str) -> dict[str, Any]:
        """Set the Lumen review status of a Label Studio task.

        Open-source Label Studio does not expose a top-level custom ``status``
        field on task updates. Store Lumen's state-machine status in task
        metadata so later sync steps can read it without relying on
        undocumented API fields.
        """
        current_resp = self._session.get(f"{self._url}/api/tasks/{task_id}/")
        current_resp.raise_for_status()
        current = current_resp.json()
        meta = dict(current.get("meta") or {})
        meta["lumen_status"] = status
        resp = self._session.patch(
            f"{self._url}/api/tasks/{task_id}/",
            json={"meta": meta},
        )
        resp.raise_for_status()
        updated = resp.json()
        updated_meta = dict(updated.get("meta") or meta)
        updated_meta["lumen_status"] = status
        updated["meta"] = updated_meta
        return updated

    def mark_reviewed(self, task_ids: list[int]) -> list[dict[str, Any]]:
        """Mark multiple tasks as reviewed (accepted)."""
        results: list[dict[str, Any]] = []
        for tid in task_ids:
            results.append(self.set_status(tid, "accepted"))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_update_config(self, project_id: int, label_config: str) -> None:
        resp = self._session.get(f"{self._url}/api/projects/{project_id}/")
        resp.raise_for_status()
        current = resp.json().get("label_config", "")
        if current != label_config:
            self._session.patch(
                f"{self._url}/api/projects/{project_id}/",
                json={"label_config": label_config},
            ).raise_for_status()


def _image_size(arr: np.ndarray) -> tuple[int, int]:
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3:
        return int(arr.shape[-2]), int(arr.shape[-1])
    raise ValueError(f"Unsupported image rank: {arr.ndim}")


def _items_from_response(body: Any) -> list[dict[str, Any]]:
    if isinstance(body, list):
        return [item for item in body if isinstance(item, dict)]
    if isinstance(body, dict) and isinstance(body.get("results"), list):
        return [item for item in body["results"] if isinstance(item, dict)]
    return []


def _next_url(body: Any, base_url: str) -> str | None:
    if not isinstance(body, dict):
        return None
    next_url = body.get("next")
    if not isinstance(next_url, str) or not next_url:
        return None
    if next_url.startswith(("http://", "https://")):
        return next_url
    return f"{base_url}{next_url if next_url.startswith('/') else f'/{next_url}'}"


def _parse_iso8601(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_optional_iso8601(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return _parse_iso8601(value)
    except ValueError:
        return None


__all__ = ["LabelStudioClient"]
