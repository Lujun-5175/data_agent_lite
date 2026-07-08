"""
Profile service — ProfileHelperAPI. Extracted from tools.py.
"""
from __future__ import annotations

from typing import Any

from src.data_manager import (
    get_analysis_preprocess_artifact,
    get_model_prep_plan,
    get_schema_profile,
)
from src.result_types import get_artifact_repository
from src.safe_executor import SafeExecutionError


class ProfileHelperAPI:
    """Schema/profile introspection API exposed to the LLM's Python execution context."""

    def __init__(self, *, dataset_id: str | None):
        self._dataset_id = dataset_id

    def schema(self) -> dict[str, Any]:
        dsid = self._require_dataset_id()
        return get_schema_profile(dsid)

    def analysis_preprocess(self) -> dict[str, Any]:
        dsid = self._require_dataset_id()
        return get_analysis_preprocess_artifact(dsid)

    def model_prep_plan(self, target: str | None = None, features: list[str] | None = None) -> dict[str, Any]:
        dsid = self._require_dataset_id()
        return get_model_prep_plan(dsid, target=target, features=features)

    def latest(self, artifact_type: str | None = None) -> dict[str, Any]:
        dsid = self._require_dataset_id()
        result = get_artifact_repository().get_latest(dsid, artifact_type=artifact_type)
        return result or {}

    def _require_dataset_id(self) -> str:
        if not self._dataset_id:
            raise SafeExecutionError("No active dataset. Please upload data first.")
        return self._dataset_id
