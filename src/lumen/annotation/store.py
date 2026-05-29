"""SQLite-backed store for tracking labelling task state."""

from __future__ import annotations

import datetime
from typing import Literal

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

TaskStatus = Literal["unlabelled", "predicted", "in_review", "accepted", "rejected"]

VALID_STATUSES = {"unlabelled", "predicted", "in_review", "accepted", "rejected"}

Base = declarative_base()


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class LabellingTask(Base):
    """SQLAlchemy model for a single labelling task."""

    __tablename__ = "labelling_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, nullable=False, index=True)
    project_id = Column(Integer, nullable=False, index=True)
    ls_task_id = Column(Integer, nullable=True, unique=True)
    status = Column(String(32), nullable=False, default="unlabelled", index=True)
    model_version = Column(String(64), nullable=True)
    prediction_hash = Column(String(64), nullable=True)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"LabellingTask(image_path={self.image_path!r}, "
            f"project_id={self.project_id}, "
            f"ls_task_id={self.ls_task_id}, "
            f"status={self.status!r})"
        )


class LabellingTaskStore:
    """SQLite-backed store for tracking labelling task lifecycle.

    Tracks the mapping between image paths, Label Studio project/task IDs,
    and review status so that pipelines can query which images need
    annotation, which have predictions, and which have been accepted.
    """

    def __init__(self, db_url: str = "sqlite:///labelling_tasks.db") -> None:
        self._engine = create_engine(db_url)
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    def _session(self) -> Session:
        return self._session_factory()

    def add_task(
        self,
        *,
        image_path: str,
        project_id: int,
        ls_task_id: int | None = None,
        status: TaskStatus = "unlabelled",
        model_version: str | None = None,
        prediction_hash: str | None = None,
    ) -> LabellingTask:
        """Insert a new labelling task record."""
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r}. Must be one of {VALID_STATUSES}")
        with self._session() as session:
            task = LabellingTask(
                image_path=image_path,
                project_id=project_id,
                ls_task_id=ls_task_id,
                status=status,
                model_version=model_version,
                prediction_hash=prediction_hash,
            )
            session.add(task)
            session.commit()
            session.expunge(task)
            return task

    def get_by_ls_task_id(self, ls_task_id: int) -> LabellingTask | None:
        with self._session() as session:
            task = (
                session.query(LabellingTask)
                .filter(LabellingTask.ls_task_id == ls_task_id)
                .one_or_none()
            )
            if task is not None:
                session.expunge(task)
            return task

    def get_by_image_path(self, image_path: str) -> list[LabellingTask]:
        with self._session() as session:
            tasks = (
                session.query(LabellingTask)
                .filter(LabellingTask.image_path == image_path)
                .all()
            )
            for t in tasks:
                session.expunge(t)
            return tasks

    def update_status(self, ls_task_id: int, status: TaskStatus) -> LabellingTask | None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r}. Must be one of {VALID_STATUSES}")
        with self._session() as session:
            task = (
                session.query(LabellingTask)
                .filter(LabellingTask.ls_task_id == ls_task_id)
                .one_or_none()
            )
            if task is None:
                return None
            task.status = status
            session.commit()
            session.expunge(task)
            return task

    def set_ls_task_id(self, image_path: str, project_id: int, ls_task_id: int) -> LabellingTask | None:
        with self._session() as session:
            task = (
                session.query(LabellingTask)
                .filter(
                    LabellingTask.image_path == image_path,
                    LabellingTask.project_id == project_id,
                )
                .one_or_none()
            )
            if task is None:
                return None
            task.ls_task_id = ls_task_id
            session.commit()
            session.expunge(task)
            return task

    def list_by_status(self, status: TaskStatus) -> list[LabellingTask]:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r}. Must be one of {VALID_STATUSES}")
        with self._session() as session:
            tasks = (
                session.query(LabellingTask)
                .filter(LabellingTask.status == status)
                .all()
            )
            for t in tasks:
                session.expunge(t)
            return tasks

    def list_by_project(self, project_id: int) -> list[LabellingTask]:
        with self._session() as session:
            tasks = (
                session.query(LabellingTask)
                .filter(LabellingTask.project_id == project_id)
                .all()
            )
            for t in tasks:
                session.expunge(t)
            return tasks


__all__ = ["LabellingTaskStore", "LabellingTask", "TaskStatus"]
