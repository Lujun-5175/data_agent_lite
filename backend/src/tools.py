from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.audit_log import get_audit_logger
from src.data_manager import (
    DatasetLoadError,
    DatasetNotFoundError,
    get_dataframe,
)
from src.errors import AppError
from src.ml_helpers import BaselineMLService, MLHelperError, MLHelperAPI
from src.request_context import get_route_diagnostics
from src.self_correction import build_repair_prompt, classify_execution_error
from src.result_types import build_artifact, get_artifact_repository
from src.settings import SETTINGS
from src.safe_executor import (
    SafeExecutionError,
    ToolExecutionTimeoutError,
    safe_execute_python,
    ReadOnlyDataFrameProxy,
    ReadOnlySeriesProxy,
    bind_current_dataset_id,
    get_current_dataset_id,
)
from src.stats_service import DataHelperAPI, StatsHelperAPI
from src.profile_service import ProfileHelperAPI

logger = logging.getLogger(__name__)


class PythonCodeInput(BaseModel):
    py_code: str = Field(description="Python code. Available variables: df, data, stats, profile, ml.")


class MLLogisticFitInput(BaseModel):
    target: str = Field(description="Binary classification target column name, e.g. Churn.")
    features: list[str] | None = Field(default=None, description="Optional feature column list.")
    test_size: float | None = Field(default=None, description="Optional test set ratio.")
    positive_label: Any | None = Field(default=None, description="Optional positive class label.")


class MLLinearRegressionFitInput(BaseModel):
    target: str = Field(description="Numeric regression target column name.")
    features: list[str] | None = Field(default=None, description="Optional feature column list.")
    test_size: float | None = Field(default=None, description="Optional test set ratio.")


class MLMetricsInput(BaseModel):
    model_artifact_id: str | None = Field(default=None, description="可选模型 artifact ID。")


class MLFeatureImportanceInput(BaseModel):
    model_artifact_id: str | None = Field(default=None, description="可选模型 artifact ID。")
    top_k: int = Field(default=10, description="Return top K important features.")


class MLLatestInput(BaseModel):
    artifact_type: str | None = Field(default=None, description="可选 artifact 类型，例如 model_result。")


class MLExecuteInput(BaseModel):
    action: Literal["train", "metrics", "feature_importance", "latest"] = Field(
        description="要执行的 ML 动作。"
    )
    model_type: Literal["logistic_regression", "linear_regression"] | None = Field(
        default=None,
        description="Model type used for the training action.",
    )
    target: str | None = Field(default=None, description="Target column for the training action.")
    features: list[str] | None = Field(default=None, description="Optional feature column list.")
    test_size: float | None = Field(default=None, description="Optional test set ratio.")
    positive_label: Any | None = Field(default=None, description="Optional positive class label.")
    model_artifact_id: str | None = Field(default=None, description="可选模型 artifact ID。")
    top_k: int = Field(default=10, description="Return top K important features.")
    artifact_type: str | None = Field(default=None, description="查询 latest 时的 artifact 类型。")


class StatsExecuteInput(BaseModel):
    action: Literal["describe_numeric", "describe_categorical", "group_summary", "correlation", "t_test", "chi_square", "anova", "latest"] = Field(
        description="Statistical action to execute."
    )
    columns: list[str] | None = Field(default=None, description="Columns needed for descriptive statistics or correlation analysis.")
    group_by: str | None = Field(default=None, description="Group column used for group summary.")
    metrics: list[dict[str, Any]] | None = Field(default=None, description="Metric definitions for group summary.")
    sort_by: str | None = Field(default=None, description="Sort column for group summary.")
    ascending: bool = Field(default=False, description="Whether to sort in ascending order.")
    top_n: int | None = Field(default=None, description="Number of top rows to return for group summary.")
    value_col: str | None = Field(default=None, description="t 检验或 ANOVA 的数值列。")
    group_col: str | None = Field(default=None, description="Group column for t-test, ANOVA, or chi-square test.")
    group_a: Any | None = Field(default=None, description="t 检验组 A 的标签。")
    group_b: Any | None = Field(default=None, description="t 检验组 B 的标签。")
    col_a: str | None = Field(default=None, description="Column A for chi-square test.")
    col_b: str | None = Field(default=None, description="Column B for chi-square test.")
    artifact_type: str | None = Field(default=None, description="查询 latest 时的 artifact 类型。")


def _get_dataset_df() -> pd.DataFrame | None:
    dataset_id = get_current_dataset_id()
    try:
        if dataset_id is None:
            return None
        return get_dataframe(dataset_id=dataset_id)
    except DatasetNotFoundError:
        logger.warning("Dataset no longer exists during tool execution", extra={"dataset_id": dataset_id})
        return None


def _build_helper_api(df: pd.DataFrame) -> tuple[DataHelperAPI, StatsHelperAPI, ProfileHelperAPI, MLHelperAPI]:
    dataset_id = get_current_dataset_id()
    return (
        DataHelperAPI(df),
        StatsHelperAPI(df, dataset_id=dataset_id),
        ProfileHelperAPI(dataset_id=dataset_id),
        MLHelperAPI(df, dataset_id=dataset_id),
    )


def _serialize_tool_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _serialize_tool_error(
    *,
    error_code: str,
    message: str,
    tool_name: str,
    retryable: bool = False,
    stage: str = "tool_execution",
) -> str:
    return _serialize_tool_payload(
        {
            "artifact_type": "tool_error",
            "error_code": error_code,
            "message": message,
            "retryable": retryable,
            "stage": stage,
            "tool_name": tool_name,
        }
    )


def _audit_tool_args(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def _output_size_bytes(result: str | None) -> int:
    if result is None:
        return 0
    return len(result.encode("utf-8"))


def _available_columns_for_self_correction(df: pd.DataFrame | None) -> list[str] | None:
    if df is None:
        return None
    return [str(column) for column in df.columns]


def _semantic_column_types_for_self_correction(dataset_id: str | None) -> dict[str, str] | None:
    if not dataset_id:
        return None
    try:
        schema_profile = get_schema_profile(dataset_id)
    except Exception:
        return None
    columns = schema_profile.get("columns", []) if isinstance(schema_profile, dict) else []
    semantic_types: dict[str, str] = {}
    for column in columns:
        if not isinstance(column, dict):
            continue
        name = column.get("column_name")
        semantic_type = column.get("semantic_type")
        if isinstance(name, str) and isinstance(semantic_type, str):
            semantic_types[name] = semantic_type
    return semantic_types or None


def _model_candidates_for_self_correction(dataset_id: str | None) -> tuple[list[str], list[str]]:
    if not dataset_id:
        return [], []
    try:
        model_prep_plan = get_model_prep_plan(dataset_id)
    except Exception:
        return [], []
    target = model_prep_plan.get("target") if isinstance(model_prep_plan, dict) else None
    candidate_features = model_prep_plan.get("candidate_features", []) if isinstance(model_prep_plan, dict) else []
    target_candidates = [target] if isinstance(target, str) and target.strip() else []
    feature_candidates = [
        str(item) for item in candidate_features if isinstance(candidate_features, list) and isinstance(item, str)
    ]
    return target_candidates, feature_candidates


def _structured_error_extra(
    error: BaseException | str,
    *,
    dataset_id: str | None = None,
    available_columns: list[str] | None = None,
    original_code: str | None = None,
    tool_name: str | None = None,
    tool_action: str | None = None,
) -> dict[str, Any] | None:
    if not SETTINGS.self_correction_enabled:
        return None

    semantic_column_types = _semantic_column_types_for_self_correction(dataset_id)
    target_candidates, feature_candidates = _model_candidates_for_self_correction(dataset_id)
    structured_error = classify_execution_error(
        error,
        available_columns=available_columns,
        semantic_column_types=semantic_column_types,
        target_candidates=target_candidates,
        feature_candidates=feature_candidates,
        tool_name=tool_name,
        tool_action=tool_action,
    )
    if original_code is not None:
        structured_error.repair_prompt = build_repair_prompt(
            original_code=original_code,
            structured_error=structured_error,
            available_columns=available_columns,
            semantic_column_types=semantic_column_types,
            tool_name=tool_name,
            tool_action=tool_action,
        )

    payload: dict[str, Any] = {
        "error_type": structured_error.error_type,
        "retryable": structured_error.retryable,
        "safe_to_retry": structured_error.safe_to_retry,
    }
    if structured_error.related_name is not None:
        payload["related_name"] = structured_error.related_name
    if structured_error.missing_column is not None:
        payload["missing_column"] = structured_error.missing_column
    if structured_error.suggestions:
        payload["suggestions"] = [
            {
                "original": suggestion.original,
                "suggestion": suggestion.suggestion,
                "score": suggestion.score,
            }
            for suggestion in structured_error.suggestions[: SETTINGS.self_correction_max_suggestions]
        ]
    if structured_error.target_candidates:
        payload["target_candidates"] = structured_error.target_candidates
    if structured_error.feature_candidates:
        payload["feature_candidates"] = structured_error.feature_candidates
    return {"structured_error": payload}


def _is_execution_failure_text(result: str | None) -> bool:
    if not isinstance(result, str):
        return False
    return result.startswith("Code execution failed:")


def _record_tool_audit(
    *,
    tool_name: str,
    dataset_id: str | None,
    tool_args: dict[str, Any] | None,
    code: str | None,
    execution_status: Literal["success", "error", "timeout", "blocked"],
    start: float,
    result: str | None = None,
    error_message: str | None = None,
    blocked_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        merged_extra = dict(extra or {})
        route_diagnostics = get_route_diagnostics()
        if isinstance(route_diagnostics, dict) and route_diagnostics:
            merged_extra["routing"] = route_diagnostics
        get_audit_logger().record(
            tool_name=tool_name,
            dataset_id=dataset_id,
            tool_args=tool_args,
            code=code,
            execution_status=execution_status,
            latency_ms=(time.perf_counter() - start) * 1000,
            output_size_bytes=_output_size_bytes(result),
            error_message=error_message,
            blocked_reason=blocked_reason,
            extra=merged_extra or None,
        )
    except Exception:
        logger.warning("Audit logging failed for tool %s", tool_name, exc_info=True)


@tool(args_schema=PythonCodeInput)
def python_inter(py_code: str) -> str:
    """
    Execute Python data analysis code in a secure sandbox.
    Available objects: `df` (pandas DataFrame, read-only), `pd`, `np`, `data`, `stats`, `profile`, `ml` helper APIs.
    `import` is BLOCKED. Use `pd.DataFrame()`, `pd.to_datetime()` directly.
    Use `print()` to output results.
    When a column name contains spaces or special chars, use df["col name"] syntax.
    Example: monthly = df.assign(period=df["date"].dt.to_period("M")).groupby("period")["value"].sum()
    """
    start = time.perf_counter()
    dataset_id = get_current_dataset_id()
    df = _get_dataset_df()
    if df is None:
        result = "错误：No active dataset. Please upload data first."
        _record_tool_audit(
            tool_name="python_inter",
            dataset_id=dataset_id,
            tool_args={},
            code=py_code,
            execution_status="success",
            start=start,
            result=result,
        )
        return result

    available_columns = _available_columns_for_self_correction(df)
    try:
        data, stats, profile, ml = _build_helper_api(df)
        env = {
            "df": ReadOnlyDataFrameProxy(df),
            "data": data,
            "stats": stats,
            "profile": profile,
            "ml": ml,
        }
        result = safe_execute_python(py_code, env, df=df)
        if _is_execution_failure_text(result):
            _record_tool_audit(
                tool_name="python_inter",
                dataset_id=dataset_id,
                tool_args={},
                code=py_code,
                execution_status="error",
                start=start,
                result=result,
                error_message=result,
                extra=_structured_error_extra(
                    result,
                    dataset_id=dataset_id,
                    available_columns=available_columns,
                    original_code=py_code,
                    tool_name="python_inter",
                ),
            )
            return result
        _record_tool_audit(
            tool_name="python_inter",
            dataset_id=dataset_id,
            tool_args={},
            code=py_code,
            execution_status="success",
            start=start,
            result=result,
        )
        return result
    except ToolExecutionTimeoutError as exc:
        result = _serialize_tool_error(
            error_code=exc.code,
            message=exc.message,
            tool_name="python_inter",
        )
        _record_tool_audit(
            tool_name="python_inter",
            dataset_id=dataset_id,
            tool_args={},
            code=py_code,
            execution_status="timeout",
            start=start,
            result=result,
            error_message=exc.message,
            extra=_structured_error_extra(
                exc,
                dataset_id=dataset_id,
                available_columns=available_columns,
                original_code=py_code,
                tool_name="python_inter",
            ),
        )
        return result
    except SafeExecutionError as exc:
        result = f"Code blocked by security policy: {exc}"
        _record_tool_audit(
            tool_name="python_inter",
            dataset_id=dataset_id,
            tool_args={},
            code=py_code,
            execution_status="blocked",
            start=start,
            result=result,
            error_message=str(exc),
            blocked_reason=str(exc),
            extra=_structured_error_extra(
                exc,
                dataset_id=dataset_id,
                available_columns=available_columns,
                original_code=py_code,
                tool_name="python_inter",
            ),
        )
        return result
    except Exception as exc:
        _record_tool_audit(
            tool_name="python_inter",
            dataset_id=dataset_id,
            tool_args={},
            code=py_code,
            execution_status="error",
            start=start,
            error_message=str(exc),
            extra=_structured_error_extra(
                exc,
                dataset_id=dataset_id,
                available_columns=available_columns,
                original_code=py_code,
                tool_name="python_inter",
            ),
        )
        raise


@tool(args_schema=MLLogisticFitInput)
def ml_logistic_fit(target: str, features: list[str] | None = None, test_size: float | None = None, positive_label: Any | None = None) -> str:
    """
    Train a baseline logistic regression model and return structured model_result.
    """
    df = _get_dataset_df()
    if df is None:
        return "错误：No active dataset. Please upload data first."

    _, _, _, ml = _build_helper_api(df)
    try:
        artifact = ml.logistic_fit(
            target=target,
            features=features,
            test_size=test_size,
            positive_label=positive_label,
        )
        return _serialize_tool_payload(artifact)
    except SafeExecutionError as exc:
        return f"错误：{exc}"


@tool(args_schema=MLLinearRegressionFitInput)
def ml_linear_regression_fit(target: str, features: list[str] | None = None, test_size: float | None = None) -> str:
    """
    Train a baseline linear regression model and return structured model_result.
    """
    df = _get_dataset_df()
    if df is None:
        return "错误：No active dataset. Please upload data first."

    _, _, _, ml = _build_helper_api(df)
    try:
        artifact = ml.linear_regression_fit(target=target, features=features, test_size=test_size)
        return _serialize_tool_payload(artifact)
    except SafeExecutionError as exc:
        return f"错误：{exc}"


@tool(args_schema=MLMetricsInput)
def ml_metrics(model_artifact_id: str | None = None) -> str:
    """
    Return metrics_result for an existing model.
    """
    df = _get_dataset_df()
    if df is None:
        return "错误：No active dataset. Please upload data first."

    _, _, _, ml = _build_helper_api(df)
    try:
        artifact = ml.metrics(model_artifact_id=model_artifact_id)
        return _serialize_tool_payload(artifact)
    except SafeExecutionError as exc:
        return f"错误：{exc}"


@tool(args_schema=MLFeatureImportanceInput)
def ml_feature_importance(model_artifact_id: str | None = None, top_k: int = 10) -> str:
    """
    Return feature_importance_result for an existing model.
    """
    df = _get_dataset_df()
    if df is None:
        return "错误：No active dataset. Please upload data first."

    _, _, _, ml = _build_helper_api(df)
    try:
        artifact = ml.feature_importance(model_artifact_id=model_artifact_id, top_k=top_k)
        return _serialize_tool_payload(artifact)
    except SafeExecutionError as exc:
        return f"错误：{exc}"


@tool(args_schema=MLLatestInput)
def ml_latest(artifact_type: str | None = None) -> str:
    """
    Return the latest ML structured result.
    """
    df = _get_dataset_df()
    if df is None:
        return "错误：No active dataset. Please upload data first."

    _, _, _, ml = _build_helper_api(df)
    try:
        artifact = ml.latest(artifact_type=artifact_type)
        return _serialize_tool_payload(artifact)
    except SafeExecutionError as exc:
        return f"错误：{exc}"


@tool(args_schema=MLExecuteInput)
def ml_execute(
    action: Literal["train", "metrics", "feature_importance", "latest"],
    model_type: Literal["logistic_regression", "linear_regression"] | None = None,
    target: str | None = None,
    features: list[str] | None = None,
    test_size: float | None = None,
    positive_label: Any | None = None,
    model_artifact_id: str | None = None,
    top_k: int = 10,
    artifact_type: str | None = None,
) -> str:
    """
    统一的 baseline ML 唯一入口。
    For ML requests, prefer this tool over python_inter.
    - action="train": Train logistic or linear regression
    - action="metrics": Return model metrics
    - action="feature_importance": Return feature importance
    - action="latest": Return the latest ML structured result
    """
    start = time.perf_counter()
    dataset_id = get_current_dataset_id()
    tool_args = _audit_tool_args(
        action=action,
        model_type=model_type,
        target=target,
        features=features,
        test_size=test_size,
        positive_label=positive_label,
        model_artifact_id=model_artifact_id,
        top_k=top_k,
        artifact_type=artifact_type,
    )
    df = _get_dataset_df()
    if df is None:
        result = "错误：No active dataset. Please upload data first."
        _record_tool_audit(
            tool_name="ml_execute",
            dataset_id=dataset_id,
            tool_args=tool_args,
            code=None,
            execution_status="success",
            start=start,
            result=result,
        )
        return result

    available_columns = _available_columns_for_self_correction(df)
    try:
        _, _, _, ml = _build_helper_api(df)
        if action == "train":
            if not target:
                result = "Error: training action must provide target."
                _record_tool_audit(
                    tool_name="ml_execute",
                    dataset_id=dataset_id,
                    tool_args=tool_args,
                    code=None,
                    execution_status="success",
                    start=start,
                    result=result,
                )
                return result
            if model_type == "linear_regression":
                artifact = ml.linear_regression_fit(
                    target=target,
                    features=features,
                    test_size=test_size,
                )
            else:
                artifact = ml.logistic_fit(
                    target=target,
                    features=features,
                    test_size=test_size,
                    positive_label=positive_label,
                )
        elif action == "metrics":
            artifact = ml.metrics(model_artifact_id=model_artifact_id)
        elif action == "feature_importance":
            artifact = ml.feature_importance(model_artifact_id=model_artifact_id, top_k=top_k)
        else:
            artifact = ml.latest(artifact_type=artifact_type)
        result = _serialize_tool_payload(artifact)
        _record_tool_audit(
            tool_name="ml_execute",
            dataset_id=dataset_id,
            tool_args=tool_args,
            code=None,
            execution_status="success",
            start=start,
            result=result,
        )
        return result
    except SafeExecutionError as exc:
        result = f"错误：{exc}"
        _record_tool_audit(
            tool_name="ml_execute",
            dataset_id=dataset_id,
            tool_args=tool_args,
            code=None,
            execution_status="error",
            start=start,
            result=result,
            error_message=str(exc),
            extra=_structured_error_extra(
                exc,
                dataset_id=dataset_id,
                available_columns=available_columns,
                tool_name="ml_execute",
                tool_action=action,
            ),
        )
        return result
    except Exception as exc:
        _record_tool_audit(
            tool_name="ml_execute",
            dataset_id=dataset_id,
            tool_args=tool_args,
            code=None,
            execution_status="error",
            start=start,
            error_message=str(exc),
            extra=_structured_error_extra(
                exc,
                dataset_id=dataset_id,
                available_columns=available_columns,
                tool_name="ml_execute",
                tool_action=action,
            ),
        )
        raise


@tool(args_schema=StatsExecuteInput)
def stats_execute(
    action: Literal["describe_numeric", "describe_categorical", "group_summary", "correlation", "t_test", "chi_square", "anova", "latest"],
    columns: list[str] | None = None,
    group_by: str | None = None,
    metrics: list[dict[str, Any]] | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
    top_n: int | None = None,
    value_col: str | None = None,
    group_col: str | None = None,
    group_a: Any | None = None,
    group_b: Any | None = None,
    col_a: str | None = None,
    col_b: str | None = None,
    artifact_type: str | None = None,
) -> str:
    """
    Unified statistical analysis entry point. Prefer this tool for statistics requests over writing custom Python code.
    """
    start = time.perf_counter()
    dataset_id = get_current_dataset_id()
    tool_args = _audit_tool_args(
        action=action,
        columns=columns,
        group_by=group_by,
        metrics=metrics,
        sort_by=sort_by,
        ascending=ascending,
        top_n=top_n,
        value_col=value_col,
        group_col=group_col,
        group_a=group_a,
        group_b=group_b,
        col_a=col_a,
        col_b=col_b,
        artifact_type=artifact_type,
    )
    df = _get_dataset_df()
    if df is None:
        result = "错误：No active dataset. Please upload data first."
        _record_tool_audit(
            tool_name="stats_execute",
            dataset_id=dataset_id,
            tool_args=tool_args,
            code=None,
            execution_status="success",
            start=start,
            result=result,
        )
        return result

    available_columns = _available_columns_for_self_correction(df)
    try:
        _, _, stats, _, _ = _build_helper_api(df)
        if action == "describe_numeric":
            artifact = stats.describe_numeric(columns)
        elif action == "describe_categorical":
            artifact = stats.describe_categorical(columns)
        elif action == "group_summary":
            if not group_by:
                result = "错误：group_summary 需要提供 group_by。"
                _record_tool_audit(
                    tool_name="stats_execute",
                    dataset_id=dataset_id,
                    tool_args=tool_args,
                    code=None,
                    execution_status="success",
                    start=start,
                    result=result,
                )
                return result
            artifact = stats.group_summary(
                group_by=group_by,
                metrics=metrics,
                sort_by=sort_by,
                ascending=ascending,
                top_n=top_n,
            )
        elif action == "correlation":
            if not columns:
                result = "Error: correlation requires at least two columns."
                _record_tool_audit(
                    tool_name="stats_execute",
                    dataset_id=dataset_id,
                    tool_args=tool_args,
                    code=None,
                    execution_status="success",
                    start=start,
                    result=result,
                )
                return result
            artifact = stats.correlation(columns)
        elif action == "t_test":
            if not value_col or not group_col or group_a is None or group_b is None:
                result = "错误：t_test 需要提供 value_col、group_col、group_a 和 group_b。"
                _record_tool_audit(
                    tool_name="stats_execute",
                    dataset_id=dataset_id,
                    tool_args=tool_args,
                    code=None,
                    execution_status="success",
                    start=start,
                    result=result,
                )
                return result
            artifact = stats.t_test(value_col, group_col, group_a, group_b)
        elif action == "chi_square":
            if not col_a or not col_b:
                result = "错误：chi_square 需要提供 col_a 和 col_b。"
                _record_tool_audit(
                    tool_name="stats_execute",
                    dataset_id=dataset_id,
                    tool_args=tool_args,
                    code=None,
                    execution_status="success",
                    start=start,
                    result=result,
                )
                return result
            artifact = stats.chi_square(col_a, col_b)
        elif action == "anova":
            if not value_col or not group_col:
                result = "错误：anova 需要提供 value_col 和 group_col。"
                _record_tool_audit(
                    tool_name="stats_execute",
                    dataset_id=dataset_id,
                    tool_args=tool_args,
                    code=None,
                    execution_status="success",
                    start=start,
                    result=result,
                )
                return result
            artifact = stats.anova(value_col, group_col)
        else:
            artifact = stats.latest(artifact_type=artifact_type)
        result = _serialize_tool_payload(artifact)
        _record_tool_audit(
            tool_name="stats_execute",
            dataset_id=dataset_id,
            tool_args=tool_args,
            code=None,
            execution_status="success",
            start=start,
            result=result,
        )
        return result
    except SafeExecutionError as exc:
        result = f"错误：{exc}"
        _record_tool_audit(
            tool_name="stats_execute",
            dataset_id=dataset_id,
            tool_args=tool_args,
            code=None,
            execution_status="error",
            start=start,
            result=result,
            error_message=str(exc),
            extra=_structured_error_extra(
                exc,
                dataset_id=dataset_id,
                available_columns=available_columns,
                tool_name="stats_execute",
                tool_action=action,
            ),
        )
        return result
    except Exception as exc:
        _record_tool_audit(
            tool_name="stats_execute",
            dataset_id=dataset_id,
            tool_args=tool_args,
            code=None,
            execution_status="error",
            start=start,
            error_message=str(exc),
            extra=_structured_error_extra(
                exc,
                dataset_id=dataset_id,
                available_columns=available_columns,
                tool_name="stats_execute",
                tool_action=action,
            ),
        )
        raise
