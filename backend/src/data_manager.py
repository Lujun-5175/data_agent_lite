from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import logging
from pathlib import Path
from threading import Event, Lock
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from src.dataset_recommendations import generate_recommended_prompts
from src.errors import AppError
from src.preprocessing import ModelPrepPlanError, plan_model_preprocessing, prepare_analysis_dataframe
from src.repositories import DatasetRepository
from src.result_types import build_artifact, get_artifact_repository
from src.schema_profile import profile_dataframe
from src.settings import SETTINGS

logger = logging.getLogger(__name__)

SUPPORTED_ENCODINGS = ("utf-8", "utf-8-sig", "gbk")
PREVIEW_ROW_COUNT = SETTINGS.preview_row_count
ARTIFACT_TTL_SECONDS = SETTINGS.artifact_ttl_seconds


class DatasetError(AppError):
    """Base dataset error."""


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset cannot be found."""

    def __init__(self, message: str = "Dataset not found or has been deleted.") -> None:
        super().__init__("dataset_not_found", message, 404)


class DatasetLoadError(DatasetError):
    """Raised when a CSV file cannot be loaded safely."""

    def __init__(self, message: str = "Failed to read CSV file. Please check the file content.") -> None:
        super().__init__("dataset_load_error", message, 400)


@dataclass(slots=True)
class Dataset:
    dataset_id: str
    raw_df: pd.DataFrame
    analysis_df: pd.DataFrame | None
    original_filename: str
    preview: list[dict[str, Any]]
    columns: list[dict[str, str]]
    created_at: datetime
    stored_path: Path
    encoding: str
    preprocessing_log: list[str]
    schema_profile_artifact: dict[str, Any]
    recommended_prompts: list[str] = field(default_factory=list)
    analysis_preprocess_artifact: dict[str, Any] | None = None
    model_prep_plan_artifact: dict[str, Any] | None = None
    preprocessed: bool = False
    generated_image_files: set[str] = field(default_factory=set)

    @property
    def row_count(self) -> int:
        return int(len(self.working_df.index))

    @property
    def original_row_count(self) -> int:
        return int(len(self.raw_df.index))

    @property
    def column_count(self) -> int:
        return int(len(self.working_df.columns))

    @property
    def preview_count(self) -> int:
        return len(self.preview)

    @property
    def analysis_basis(self) -> str:
        return "analysis_df" if self.analysis_df is not None else "raw_df"

    @property
    def original_df(self) -> pd.DataFrame:
        return self.raw_df

    @property
    def working_df(self) -> pd.DataFrame:
        return self.analysis_df if self.analysis_df is not None else self.raw_df

    @working_df.setter
    def working_df(self, value: pd.DataFrame) -> None:
        self.analysis_df = value

    def register_generated_image(self, filename: str) -> None:
        if filename:
            self.generated_image_files.add(filename)

    def cleanup_artifacts(self) -> None:
        try:
            self.stored_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to delete stored CSV", extra={"dataset_id": self.dataset_id})

        images_dir = self.stored_path.parent.parent / "static" / "images"
        for filename in list(self.generated_image_files):
            image_path = (images_dir / filename).resolve()
            try:
                image_path.unlink(missing_ok=True)
            except Exception:
                logger.warning(
                    "Failed to delete generated image",
                    extra={"dataset_id": self.dataset_id, "filename": filename},
                )


@dataclass(slots=True)
class InMemoryDatasetRepository:
    _datasets: dict[str, Dataset] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)
    _preprocess_events: dict[str, Event] = field(default_factory=dict)
    _preprocess_errors: dict[str, Exception] = field(default_factory=dict)

    def create_dataset(
        self,
        *,
        original_df: pd.DataFrame,
        original_filename: str,
        stored_path: Path,
        encoding: str,
        preprocessing_log: list[str],
    ) -> Dataset:
        # Upload path keeps minimal processing: analysis dataframe is prepared lazily.
        dataset_id = str(uuid4())
        profile_payload = profile_dataframe(original_df)
        profile_artifact = build_artifact(
            artifact_type="schema_profile",
            dataset_id=dataset_id,
            payload={k: v for k, v in profile_payload.items() if k != "warnings"},
            warnings=list(profile_payload.get("warnings", [])),
        )
        get_artifact_repository().register(dataset_id, profile_artifact)
        dataset = Dataset(
            dataset_id=dataset_id,
            raw_df=original_df.copy(deep=True),
            analysis_df=None,
            original_filename=original_filename,
            preview=_build_preview(original_df),
            columns=_build_columns(original_df),
            created_at=datetime.now(timezone.utc),
            stored_path=stored_path,
            encoding=encoding,
            preprocessing_log=preprocessing_log,
            schema_profile_artifact=profile_artifact,
            recommended_prompts=[],
            preprocessed=False,
        )
        with self._lock:
            self._datasets[dataset.dataset_id] = dataset
        logger.info("Dataset created", extra={"dataset_id": dataset.dataset_id, "original_filename": original_filename})
        return dataset

    def get_dataset(self, dataset_id: str) -> Dataset:
        with self._lock:
            dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError("Dataset not found or has been deleted.")
        return dataset

    def delete_dataset(self, dataset_id: str) -> Dataset:
        with self._lock:
            dataset = self._datasets.pop(dataset_id, None)
            if dataset is None:
                raise DatasetNotFoundError("Dataset not found or has been deleted.")
        logger.info("Dataset deleted", extra={"dataset_id": dataset_id})
        return dataset

    def register_generated_image(self, dataset_id: str, filename: str) -> None:
        if not filename:
            return
        with self._lock:
            dataset = self._datasets.get(dataset_id)
            if dataset is None:
                return
            dataset.register_generated_image(filename)

    def list_datasets(self) -> list[Dataset]:
        with self._lock:
            return list(self._datasets.values())

    def ensure_preprocessed(self, dataset_id: str) -> Dataset:
        while True:
            with self._lock:
                dataset = self._datasets.get(dataset_id)
                if dataset is None:
                    raise DatasetNotFoundError("Dataset not found or has been deleted.")
                if dataset.preprocessed:
                    return dataset

                preprocess_event = self._preprocess_events.get(dataset_id)
                if preprocess_event is None:
                    preprocess_event = Event()
                    self._preprocess_events[dataset_id] = preprocess_event
                    self._preprocess_errors.pop(dataset_id, None)
                    raw_df = dataset.raw_df
                    schema_profile_artifact = dataset.schema_profile_artifact
                    is_owner = True
                else:
                    is_owner = False

            if not is_owner:
                preprocess_event.wait()
                with self._lock:
                    dataset = self._datasets.get(dataset_id)
                    if dataset is None:
                        raise DatasetNotFoundError("Dataset not found or has been deleted.")
                    if dataset.preprocessed:
                        return dataset
                    error = self._preprocess_errors.get(dataset_id)
                if error is not None:
                    raise error
                continue

            try:
                analysis_df, preprocess_payload, preprocess_warnings = prepare_analysis_dataframe(
                    raw_df,
                    schema_profile_artifact,
                )
                preprocess_artifact = build_artifact(
                    artifact_type="preprocess_result",
                    dataset_id=dataset_id,
                    payload=preprocess_payload,
                    warnings=preprocess_warnings,
                )
            except Exception as exc:
                with self._lock:
                    self._preprocess_errors[dataset_id] = exc
                    done_event = self._preprocess_events.pop(dataset_id, None)
                if done_event is not None:
                    done_event.set()
                raise

            with self._lock:
                dataset = self._datasets.get(dataset_id)
                if dataset is None:
                    self._preprocess_errors.pop(dataset_id, None)
                    done_event = self._preprocess_events.pop(dataset_id, None)
                    if done_event is not None:
                        done_event.set()
                    raise DatasetNotFoundError("Dataset not found or has been deleted.")

                if not dataset.preprocessed:
                    get_artifact_repository().register(dataset.dataset_id, preprocess_artifact)
                    dataset.analysis_df = analysis_df
                    dataset.analysis_preprocess_artifact = preprocess_artifact
                    dataset.preview = _build_preview(dataset.working_df)
                    dataset.columns = _build_columns(dataset.working_df)
                    dataset.preprocessing_log = _build_preprocessing_log(preprocess_artifact)
                    dataset.preprocessed = True
                    logger.info("Dataset preprocessing completed", extra={"dataset_id": dataset_id})

                self._preprocess_errors.pop(dataset_id, None)
                done_event = self._preprocess_events.pop(dataset_id, None)

            if done_event is not None:
                done_event.set()
            return dataset

    def get_schema_profile(self, dataset_id: str) -> dict[str, Any]:
        dataset = self.get_dataset(dataset_id)
        return dataset.schema_profile_artifact

    def get_analysis_preprocess_artifact(self, dataset_id: str) -> dict[str, Any]:
        dataset = self.ensure_preprocessed(dataset_id)
        if dataset.analysis_preprocess_artifact is None:
            raise DatasetLoadError("Analysis preprocessing info not available。")
        return dataset.analysis_preprocess_artifact

    def get_or_create_model_prep_plan(
        self,
        dataset_id: str,
        *,
        target: str | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        dataset = self.get_dataset(dataset_id)
        if target is None and features is None and dataset.model_prep_plan_artifact is not None:
            return dataset.model_prep_plan_artifact

        try:
            plan_payload, warnings = plan_model_preprocessing(
                dataset.raw_df,
                dataset.schema_profile_artifact,
                target=target,
                features=features,
            )
        except ModelPrepPlanError as exc:
            raise DatasetLoadError(str(exc)) from exc

        plan_artifact = build_artifact(
            artifact_type="model_prep_plan",
            dataset_id=dataset.dataset_id,
            payload=plan_payload,
            warnings=warnings,
        )
        get_artifact_repository().register(dataset.dataset_id, plan_artifact)
        dataset.model_prep_plan_artifact = plan_artifact
        return plan_artifact

    def get_or_create_recommended_prompts(self, dataset_id: str) -> list[str]:
        dataset = self.get_dataset(dataset_id)
        if dataset.recommended_prompts:
            return list(dataset.recommended_prompts)

        prompts = generate_recommended_prompts(
            dataset_summary=_build_dataset_summary(dataset),
            schema_profile=dataset.schema_profile_artifact,
        )
        with self._lock:
            current_dataset = self._datasets.get(dataset_id)
            if current_dataset is not None and not current_dataset.recommended_prompts:
                current_dataset.recommended_prompts = list(prompts)
            dataset = current_dataset or dataset
        return list(dataset.recommended_prompts)


_dataset_repository: DatasetRepository = InMemoryDatasetRepository()


def get_dataset_repository() -> DatasetRepository:
    return _dataset_repository


def set_dataset_repository(repository: DatasetRepository) -> None:
    global _dataset_repository
    _dataset_repository = repository


dataset_store = get_dataset_repository()


def _build_preprocessing_log(preprocess_artifact: dict[str, Any]) -> list[str]:
    payload = preprocess_artifact
    steps = payload.get("steps", [])
    warnings = payload.get("warnings", [])
    log_entries: list[str] = []
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("type", "unknown"))
        columns = step.get("columns")
        if isinstance(columns, list) and columns:
            col_list = ", ".join(str(item) for item in columns)
            log_entries.append(f"{step_type}: {col_list}")
        elif "reason" in step:
            log_entries.append(f"{step_type}: {step.get('reason')}")
        else:
            log_entries.append(step_type)
    for warning in warnings if isinstance(warnings, list) else []:
        log_entries.append(f"warning: {warning}")
    if not log_entries:
        log_entries.append("analysis No additional preprocessing performed at this stage。")
    return log_entries


def _build_preview(df: pd.DataFrame, count: int = PREVIEW_ROW_COUNT) -> list[dict[str, Any]]:
    return df.head(count).replace({np.nan: None}).to_dict(orient="records")


def _build_columns(df: pd.DataFrame) -> list[dict[str, str]]:
    columns: list[dict[str, str]] = []
    for name in df.columns:
        column_type = "numerical" if pd.api.types.is_numeric_dtype(df[name]) else "categorical"
        columns.append({"name": str(name), "type": column_type})
    return columns


def load_csv_file(file_path: Path, original_filename: str) -> Dataset:
    resolved_path = file_path.resolve(strict=True)

    for encoding in SUPPORTED_ENCODINGS:
        try:
            raw_df = pd.read_csv(resolved_path, encoding=encoding)
            dataset = get_dataset_repository().create_dataset(
                original_df=raw_df,
                original_filename=original_filename,
                stored_path=resolved_path,
                encoding=encoding,
                preprocessing_log=["Upload stage: basic read only, analysis_df not yet generated."],
            )
            return dataset
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise DatasetLoadError("CSV 文件为空，无法加载。") from exc
        except pd.errors.ParserError as exc:
            raise DatasetLoadError("Invalid CSV format, cannot parse.") from exc
        except Exception as exc:
            logger.exception("Unexpected error while loading CSV")
            raise DatasetLoadError("CSV File read failed. Please check the file content.") from exc

    raise DatasetLoadError(
        f"CSV encoding not supported. Please use {', '.join(SUPPORTED_ENCODINGS)} and try again."
    )


def get_dataset(dataset_id: str) -> Dataset:
    return get_dataset_repository().get_dataset(dataset_id)


def get_dataframe(dataset_id: str) -> pd.DataFrame:
    return get_dataset_repository().ensure_preprocessed(dataset_id).working_df


def get_data_preview(dataset_id: str, n: int = PREVIEW_ROW_COUNT) -> list[dict[str, Any]]:
    dataset = get_dataset_repository().ensure_preprocessed(dataset_id)
    return dataset.working_df.head(n).replace({np.nan: None}).to_dict(orient="records")


def get_data_info(dataset_id: str) -> str:
    dataset = get_dataset_repository().ensure_preprocessed(dataset_id)
    buffer = io.StringIO()
    dataset.working_df.info(buf=buffer)
    preprocessing_lines = "\n".join(f"- {entry}" for entry in dataset.preprocessing_log)
    return (
        f"Source file: {dataset.original_filename}\n"
        f"dataset_id: {dataset.dataset_id}\n"
        f"Encoding: {dataset.encoding}\n"
        f"Original shape: {dataset.original_row_count} rows x {len(dataset.raw_df.columns)} cols\n"
        f"Analysis shape: {dataset.row_count} rows x {dataset.column_count} cols\n"
        f"Analysis basis: {dataset.analysis_basis}\n"
        f"Preprocessing log:\n{preprocessing_lines}\n"
        f"{'-' * 30}\n"
        f"{buffer.getvalue()}"
    )


def get_data_context_summary(dataset_id: str) -> dict[str, Any]:
    dataset = get_dataset_repository().ensure_preprocessed(dataset_id)
    numeric_columns = [column["name"] for column in dataset.columns if column.get("type") == "numerical"]
    categorical_columns = [column["name"] for column in dataset.columns if column.get("type") != "numerical"]
    schema_warnings = dataset.schema_profile_artifact.get("warnings", [])
    preprocess_warnings = (
        dataset.analysis_preprocess_artifact.get("warnings", [])
        if isinstance(dataset.analysis_preprocess_artifact, dict)
        else []
    )
    return {
        "dataset_id": dataset.dataset_id,
        "filename": dataset.original_filename,
        "analysis_basis": dataset.analysis_basis,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "numeric_column_count": len(numeric_columns),
        "categorical_column_count": len(categorical_columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "preprocessing_log": dataset.preprocessing_log,
        "warnings": [str(item) for item in [*schema_warnings, *preprocess_warnings] if item],
    }


def _build_dataset_summary(dataset: Dataset) -> dict[str, Any]:
    working_df = dataset.working_df
    columns = _build_columns(working_df)
    numeric_columns = [column["name"] for column in columns if column.get("type") == "numerical"]
    categorical_columns = [column["name"] for column in columns if column.get("type") != "numerical"]
    schema_warnings = dataset.schema_profile_artifact.get("warnings", [])
    preprocess_warnings = (
        dataset.analysis_preprocess_artifact.get("warnings", [])
        if isinstance(dataset.analysis_preprocess_artifact, dict)
        else []
    )
    return {
        "dataset_id": dataset.dataset_id,
        "filename": dataset.original_filename,
        "analysis_basis": dataset.analysis_basis,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "warnings": [str(item) for item in [*schema_warnings, *preprocess_warnings] if item],
    }


def get_schema_profile(dataset_id: str) -> dict[str, Any]:
    return get_dataset_repository().get_schema_profile(dataset_id)


def get_analysis_preprocess_artifact(dataset_id: str) -> dict[str, Any]:
    return get_dataset_repository().get_analysis_preprocess_artifact(dataset_id)


def get_model_prep_plan(
    dataset_id: str,
    *,
    target: str | None = None,
    features: list[str] | None = None,
) -> dict[str, Any]:
    return get_dataset_repository().get_or_create_model_prep_plan(dataset_id, target=target, features=features)


def get_dataset_recommended_prompts(dataset_id: str) -> list[str]:
    return get_dataset_repository().get_or_create_recommended_prompts(dataset_id)


def calculate_correlation(dataset_id: str, col1: str, col2: str) -> dict[str, Any]:
    dataset = get_dataset_repository().ensure_preprocessed(dataset_id)
    df = dataset.working_df

    if col1 not in df.columns or col2 not in df.columns:
        raise DatasetLoadError("Specified column does not exist.")

    series1 = df[col1]
    series2 = df[col2]
    is_numeric_1 = pd.api.types.is_numeric_dtype(series1)
    is_numeric_2 = pd.api.types.is_numeric_dtype(series2)

    result: dict[str, Any] = {
        "dataset_id": dataset_id,
        "column1": col1,
        "column2": col2,
        "correlation_type": "pearson" if is_numeric_1 and is_numeric_2 else "unsupported",
        "value": None,
        "interpretation": "This type of correlation analysis is not supported in the current version.",
    }

    if not (is_numeric_1 and is_numeric_2):
        return result

    corr = series1.corr(series2, method="pearson")
    if pd.isna(corr):
        corr = 0.0

    corr_value = round(float(corr), 4)
    result["value"] = corr_value
    result["interpretation"] = _interpret_correlation(corr_value)
    return result


def _interpret_correlation(value: float) -> str:
    absolute = abs(value)
    if absolute < 0.2:
        strength = "弱"
    elif absolute < 0.4:
        strength = "较弱"
    elif absolute < 0.6:
        strength = "中等"
    elif absolute < 0.8:
        strength = "较强"
    else:
        strength = "强"

    if value > 0:
        direction = "正相关"
    elif value < 0:
        direction = "负相关"
    else:
        direction = "No significant linear correlation"

    if value == 0:
        return direction
    return f"{strength}{direction}"


def cleanup_dataset_artifacts(dataset_id: str) -> None:
    dataset = get_dataset_repository().delete_dataset(dataset_id)
    get_artifact_repository().clear_dataset(dataset.dataset_id)
    dataset.cleanup_artifacts()


def register_dataset_generated_image(dataset_id: str, filename: str) -> None:
    get_dataset_repository().register_generated_image(dataset_id, filename)


def cleanup_expired_artifacts(
    *,
    temp_dir: Path,
    images_dir: Path,
    ttl_seconds: int = ARTIFACT_TTL_SECONDS,
) -> dict[str, int]:
    cutoff = datetime.now(timezone.utc).timestamp() - ttl_seconds
    removed_temp = _cleanup_expired_files(temp_dir, cutoff=cutoff)
    removed_images = _cleanup_expired_files(images_dir, cutoff=cutoff)
    return {"temp_data": removed_temp, "images": removed_images}


def _cleanup_expired_files(directory: Path, *, cutoff: float) -> int:
    if not directory.exists():
        return 0

    removed = 0
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        except FileNotFoundError:
            continue
        except Exception:
            logger.warning("Failed to clean expired artifact", extra={"path": str(path)})
    return removed
