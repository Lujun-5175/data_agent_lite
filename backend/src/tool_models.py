"""Pydantic input models for LangChain tools. Extracted from tools.py."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PythonCodeInput(BaseModel):
    py_code: str = Field(description="Python code. Available variables: df, data, viz, stats, profile, ml.")


class FigCodeInput(BaseModel):
    py_code: str = Field(description="Plotting code. Must generate an image object.")
    fname: str = Field(description="Image variable name, e.g. 'fig'.")


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
    model_artifact_id: str | None = Field(default=None, description="Model artifact ID to evaluate. Uses latest if not provided.")


class MLFeatureImportanceInput(BaseModel):
    model_artifact_id: str | None = Field(default=None, description="Model artifact ID. Uses latest if not provided.")
    top_k: int = Field(default=10, description="Return top K important features.")


class MLLatestInput(BaseModel):
    artifact_type: str | None = Field(default=None, description="Artifact type filter (e.g. model_result).")


class MLExecuteInput(BaseModel):
    action: str = Field(description="Model type used for the training action.")
    target: str | None = Field(default=None, description="Target column for the training action.")
    features: list[str] | None = Field(default=None, description="Optional feature column list.")
    test_size: float | None = Field(default=None, description="Optional test set ratio.")
    positive_label: Any | None = Field(default=None, description="Optional positive class label.")
    top_k: int = Field(default=10, description="Top K for feature importance.")
    model_artifact_id: str | None = Field(default=None, description="Model artifact ID for metrics/importance.")
    fit_target: str | None = Field(default=None, description="Deprecated: use target instead.")
    fit_features: list[str] | None = Field(default=None, description="Deprecated: use features instead.")


class StatsExecuteInput(BaseModel):
    action: str = Field(description="Statistical action to execute.")
    columns: list[str] | None = Field(default=None, description="Columns needed for descriptive statistics or correlation analysis.")
    group_by: str | None = Field(default=None, description="Group column used for group summary.")
    metrics: list[dict[str, Any]] | None = Field(default=None, description="Metric definitions for group summary.")
    sort_by: str | None = Field(default=None, description="Sort column for group summary.")
    ascending: bool = Field(default=False, description="Whether to sort in ascending order.")
    top_n: int | None = Field(default=None, description="Number of top rows to return for group summary.")
    group_col: str | None = Field(default=None, description="Group column for t-test, ANOVA, or chi-square test.")
    group_a: Any | None = Field(default=None, description="First group for t-test.")
    group_b: Any | None = Field(default=None, description="Second group for t-test.")
    value_col: str | None = Field(default=None, description="Value column for t-test.")
    col_a: str | None = Field(default=None, description="Column A for chi-square test.")
    col_b: str | None = Field(default=None, description="Column B for chi-square test.")
    top_k: int = Field(default=10, description="Top K for correlation pairs.")
