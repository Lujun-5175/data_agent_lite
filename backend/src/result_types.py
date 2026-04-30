from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_artifact(
    *,
    artifact_type: str,
    dataset_id: str | None,
    payload: dict[str, Any],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "artifact_type": artifact_type,
        "artifact_id": str(uuid4()),
        "created_at": now_iso(),
        "dataset_id": dataset_id,
        "warnings": warnings or [],
    }
    base.update(payload)
    return base


@dataclass(slots=True)
class ArtifactRegistry:
    _latest_by_dataset: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    _artifacts_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def register(self, dataset_id: str | None, artifact: dict[str, Any]) -> dict[str, Any]:
        if not dataset_id:
            return artifact
        artifact_type = str(artifact.get("artifact_type", "unknown"))
        artifact_id = str(artifact.get("artifact_id", ""))
        with self._lock:
            dataset_bucket = self._latest_by_dataset.setdefault(dataset_id, {})
            dataset_bucket[artifact_type] = artifact
            dataset_bucket["latest"] = artifact
            if artifact_id:
                self._artifacts_by_id[artifact_id] = artifact
        return artifact

    def get_latest(self, dataset_id: str | None, artifact_type: str | None = None) -> dict[str, Any] | None:
        if not dataset_id:
            return None
        with self._lock:
            dataset_bucket = self._latest_by_dataset.get(dataset_id, {})
            if artifact_type:
                return dataset_bucket.get(artifact_type)
            return dataset_bucket.get("latest")

    def get_by_artifact_id(self, artifact_id: str | None) -> dict[str, Any] | None:
        if not artifact_id:
            return None
        with self._lock:
            return self._artifacts_by_id.get(artifact_id)

    def clear_dataset(self, dataset_id: str | None) -> None:
        if not dataset_id:
            return
        with self._lock:
            bucket = self._latest_by_dataset.pop(dataset_id, {})
            for artifact in bucket.values():
                if not isinstance(artifact, dict):
                    continue
                artifact_id = artifact.get("artifact_id")
                if isinstance(artifact_id, str):
                    self._artifacts_by_id.pop(artifact_id, None)


artifact_registry = ArtifactRegistry()
