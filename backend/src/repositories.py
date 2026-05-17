from __future__ import annotations

from typing import Any, Protocol


class DatasetRepository(Protocol):
    def create_dataset(self, **kwargs: Any) -> Any:
        ...

    def get_dataset(self, dataset_id: str) -> Any:
        ...

    def delete_dataset(self, dataset_id: str) -> Any:
        ...

    def list_datasets(self) -> list[Any]:
        ...

    def ensure_preprocessed(self, dataset_id: str) -> Any:
        ...

    def register_generated_image(self, dataset_id: str, filename: str) -> None:
        ...

    def get_schema_profile(self, dataset_id: str) -> dict[str, Any]:
        ...

    def get_analysis_preprocess_artifact(self, dataset_id: str) -> dict[str, Any]:
        ...

    def get_or_create_model_prep_plan(
        self,
        dataset_id: str,
        *,
        target: str | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        ...

    def get_or_create_recommended_prompts(self, dataset_id: str) -> list[str]:
        ...


class ArtifactRepository(Protocol):
    def register(self, dataset_id: str | None, artifact: dict[str, Any]) -> dict[str, Any]:
        ...

    def get_latest(self, dataset_id: str | None, artifact_type: str | None = None) -> dict[str, Any] | None:
        ...

    def get_by_artifact_id(self, artifact_id: str | None) -> dict[str, Any] | None:
        ...

    def clear_dataset(self, dataset_id: str | None) -> None:
        ...
