from __future__ import annotations

import ast
import contextlib
import json
import logging
import platform
import sys
import time
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.audit_log import get_audit_logger
from src.data_manager import (
    DatasetLoadError,
    DatasetNotFoundError,
    get_analysis_preprocess_artifact,
    get_dataframe,
    get_model_prep_plan,
    get_schema_profile,
    register_dataset_generated_image,
)
from src.errors import AppError
from src.request_context import get_route_diagnostics
from src.ml_helpers import BaselineMLService, MLHelperError
from src.self_correction import build_repair_prompt, classify_execution_error
from src.result_types import build_artifact, get_artifact_repository
from src.settings import SETTINGS

logger = logging.getLogger(__name__)

CURRENT_DATASET_ID: ContextVar[str | None] = ContextVar("current_dataset_id", default=None)
CURRENT_IMAGE_EVENT_STATE: ContextVar[dict[str, Any] | None] = ContextVar("current_image_event_state", default=None)

ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

FORBIDDEN_CALL_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}

FORBIDDEN_METHOD_NAMES = {
    "boxplot",
    "dump",
    "dumps",
    "export",
    "from_file",
    "from_url",
    "imsave",
    "load",
    "loads",
    "read",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_stata",
    "read_table",
    "read_xml",
    "plot",
    "save",
    "savefig",
    "to_excel",
    "to_clipboard",
    "to_csv",
    "to_feather",
    "to_gbq",
    "to_hdf",
    "to_html",
    "to_json",
    "to_latex",
    "to_markdown",
    "to_orc",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "to_stata",
    "to_xml",
    "tofile",
    "write",
    "write_html",
    "write_image",
}

FORBIDDEN_IDENTIFIERS = {
    "matplotlib",
    "np",
    "os",
    "pd",
    "pathlib",
    "plt",
    "requests",
    "seaborn",
    "shutil",
    "socket",
    "subprocess",
    "sys",
}


class SafeExecutionError(AppError):
    """Raised when code fails static validation or safe execution."""

    def __init__(self, message: str) -> None:
        super().__init__("invalid_python_code", message, 400)


class ToolExecutionTimeoutError(SafeExecutionError):
    """Raised when constrained execution exceeds its adaptive timeout budget."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"代码执行超时（>{seconds:.1f} 秒），请缩小范围、指定列或减少循环。")
        self.code = "tool_execution_timeout"


class SafeCodeValidator(ast.NodeVisitor):
    def __init__(
        self,
        *,
        max_ast_nodes: int | None = None,
        max_loop_nesting: int | None = None,
        max_comprehension_nesting: int | None = None,
        max_call_chain_depth: int | None = None,
    ) -> None:
        self.max_ast_nodes = SETTINGS.safe_exec_max_ast_nodes if max_ast_nodes is None else max_ast_nodes
        self.max_loop_nesting = SETTINGS.safe_exec_max_loop_nesting if max_loop_nesting is None else max_loop_nesting
        self.max_comprehension_nesting = (
            SETTINGS.safe_exec_max_comprehension_nesting
            if max_comprehension_nesting is None
            else max_comprehension_nesting
        )
        self.max_call_chain_depth = SETTINGS.safe_exec_max_call_chain_depth if max_call_chain_depth is None else max_call_chain_depth
        self._ast_node_count = 0
        self._loop_nesting = 0
        self._comprehension_nesting = 0

    def visit(self, node: ast.AST) -> Any:
        self._ast_node_count += 1
        if self._ast_node_count > self.max_ast_nodes:
            raise SafeExecutionError("代码过于复杂：AST 节点数超过限制。")
        return super().visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        raise SafeExecutionError("不允许在执行代码中使用 import。")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        raise SafeExecutionError("不允许在执行代码中使用 from ... import ...。")

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id.startswith("__"):
            raise SafeExecutionError(f"不允许访问敏感名称: {node.id}")
        if node.id in FORBIDDEN_IDENTIFIERS:
            raise SafeExecutionError(f"不允许访问危险标识符: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        self._check_call_chain_depth(node)
        if node.attr.startswith("_"):
            raise SafeExecutionError(f"不允许访问敏感属性: {node.attr}")
        if node.attr in FORBIDDEN_METHOD_NAMES:
            raise SafeExecutionError(f"不允许访问危险函数: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        self._check_call_chain_depth(node)
        call_name = self._resolve_call_name(node.func)
        if call_name in FORBIDDEN_CALL_NAMES or call_name in FORBIDDEN_METHOD_NAMES:
            raise SafeExecutionError(f"不允许调用危险函数: {call_name}")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        self._visit_loop_node(node)

    def visit_While(self, node: ast.While) -> Any:
        self._visit_loop_node(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        self._visit_loop_node(node)

    def visit_ListComp(self, node: ast.ListComp) -> Any:
        self._visit_comprehension_node(node)

    def visit_SetComp(self, node: ast.SetComp) -> Any:
        self._visit_comprehension_node(node)

    def visit_DictComp(self, node: ast.DictComp) -> Any:
        self._visit_comprehension_node(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        self._visit_comprehension_node(node)

    def _resolve_call_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _visit_loop_node(self, node: ast.AST) -> Any:
        self._loop_nesting += 1
        try:
            if self._loop_nesting > self.max_loop_nesting:
                raise SafeExecutionError("代码过于复杂：循环嵌套层数超过限制。")
            self.generic_visit(node)
        finally:
            self._loop_nesting -= 1

    def _visit_comprehension_node(self, node: ast.AST) -> Any:
        self._comprehension_nesting += 1
        try:
            if self._comprehension_nesting > self.max_comprehension_nesting:
                raise SafeExecutionError("代码过于复杂：推导式嵌套层数超过限制。")
            self.generic_visit(node)
        finally:
            self._comprehension_nesting -= 1

    def _check_call_chain_depth(self, node: ast.AST) -> None:
        if self._measure_chain_depth(node) > self.max_call_chain_depth:
            raise SafeExecutionError("代码过于复杂：调用链过长。")

    def _measure_chain_depth(self, node: ast.AST) -> int:
        if isinstance(node, ast.Call):
            return 1 + self._measure_chain_depth(node.func)
        if isinstance(node, ast.Attribute):
            return 1 + self._measure_chain_depth(node.value)
        return 0


class DataHelperAPI:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def columns(self) -> list[str]:
        return [str(column) for column in self._df.columns]

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._df.shape)

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._df.head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        return self._df.tail(n)

    def describe(self) -> pd.DataFrame:
        return self._df.describe(include="all")

    def numeric_summary(self) -> pd.DataFrame:
        numeric_df = self._df.select_dtypes(include=[np.number])
        return numeric_df.describe().T if not numeric_df.empty else pd.DataFrame()

    def missing_summary(self) -> pd.DataFrame:
        total_rows = max(len(self._df.index), 1)
        missing_count = self._df.isna().sum()
        missing_rate = (missing_count / total_rows).round(4)
        return pd.DataFrame(
            {
                "column": missing_count.index.astype(str),
                "missing_count": missing_count.values,
                "missing_rate": missing_rate.values,
            }
        )

    def value_counts(self, column: str, top_n: int = 10) -> pd.DataFrame:
        if column not in self._df.columns:
            raise SafeExecutionError(f"列不存在: {column}")
        counts = self._df[column].value_counts(dropna=False).head(top_n)
        return counts.reset_index().rename(columns={"index": column, column: "count"})

    def unique(self, column: str) -> list[Any]:
        if column not in self._df.columns:
            raise SafeExecutionError(f"列不存在: {column}")
        return list(self._df[column].drop_duplicates().tolist())

    def select(self, columns: list[str]) -> pd.DataFrame:
        return self._df.loc[:, columns]

    def filter_equals(self, column: str, value: Any) -> pd.DataFrame:
        if column not in self._df.columns:
            raise SafeExecutionError(f"列不存在: {column}")
        return self._df[self._df[column] == value]

    def top_rows(self, column: str, n: int = 5, ascending: bool = False) -> pd.DataFrame:
        if column not in self._df.columns:
            raise SafeExecutionError(f"列不存在: {column}")
        return self._df.sort_values(column, ascending=ascending).head(n)

    def group_mean(self, group_column: str, value_column: str) -> pd.DataFrame:
        if group_column not in self._df.columns or value_column not in self._df.columns:
            raise SafeExecutionError(f"列不存在: {group_column} / {value_column}")
        return (
            self._df.groupby(group_column, dropna=False)[value_column]
            .mean()
            .reset_index(name=f"{value_column}_mean")
        )

    def group_sum(self, group_column: str, value_column: str) -> pd.DataFrame:
        if group_column not in self._df.columns or value_column not in self._df.columns:
            raise SafeExecutionError(f"列不存在: {group_column} / {value_column}")
        return (
            self._df.groupby(group_column, dropna=False)[value_column]
            .sum()
            .reset_index(name=f"{value_column}_sum")
        )

    def correlation(self, col1: str, col2: str) -> float:
        if col1 not in self._df.columns or col2 not in self._df.columns:
            raise SafeExecutionError(f"列不存在: {col1} / {col2}")
        value = self._df[col1].corr(self._df[col2])
        return 0.0 if pd.isna(value) else float(value)


class PlotHelperAPI:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def hist(self, column: str, bins: int = 30, title: str | None = None):
        if column not in self._df.columns:
            raise SafeExecutionError(f"列不存在: {column}")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(data=self._df, x=column, bins=bins, ax=ax, color="#4F86C6")
        ax.set_title(title or f"{column} 分布")
        ax.set_xlabel(column)
        ax.set_ylabel("频数")
        fig.tight_layout()
        return fig

    def bar(self, x: str, y: str | None = None, title: str | None = None, top_n: int = 10):
        if x not in self._df.columns:
            raise SafeExecutionError(f"列不存在: {x}")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if y and y in self._df.columns:
            summary = self._df.groupby(x, dropna=False)[y].mean().reset_index(name=y)
            sns.barplot(data=summary.head(top_n), x=x, y=y, ax=ax, color="#4F86C6")
            ax.set_ylabel(y)
        else:
            counts = self._df[x].value_counts(dropna=False).head(top_n).reset_index()
            counts.columns = [x, "count"]
            sns.barplot(data=counts, x=x, y="count", ax=ax, color="#4F86C6")
            ax.set_ylabel("count")
        ax.set_title(title or f"{x} 条形图")
        ax.set_xlabel(x)
        fig.tight_layout()
        return fig

    def line(self, x: str, y: str, title: str | None = None):
        if x not in self._df.columns or y not in self._df.columns:
            raise SafeExecutionError(f"列不存在: {x} / {y}")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.lineplot(data=self._df, x=x, y=y, ax=ax, color="#4F86C6")
        ax.set_title(title or f"{x} vs {y}")
        fig.tight_layout()
        return fig

    def scatter(self, x: str, y: str, hue: str | None = None, title: str | None = None):
        if x not in self._df.columns or y not in self._df.columns:
            raise SafeExecutionError(f"列不存在: {x} / {y}")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.scatterplot(data=self._df, x=x, y=y, hue=hue if hue in self._df.columns else None, ax=ax)
        ax.set_title(title or f"{x} vs {y}")
        fig.tight_layout()
        return fig

    def box(self, y: str, x: str | None = None, title: str | None = None):
        if y not in self._df.columns:
            raise SafeExecutionError(f"列不存在: {y}")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(data=self._df, x=x if x in self._df.columns else None, y=y, ax=ax)
        ax.set_title(title or f"{y} 箱线图")
        fig.tight_layout()
        return fig

    def heatmap_corr(self, columns: list[str] | None = None, title: str | None = None):
        if columns:
            target_df = self._df.loc[:, [column for column in columns if column in self._df.columns]]
        else:
            target_df = self._df.select_dtypes(include=[np.number])
        if target_df.empty:
            raise SafeExecutionError("没有可用于相关性热力图的数值列。")
        corr_df = target_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_df, cmap="Blues", annot=True, fmt=".2f", ax=ax)
        ax.set_title(title or "相关性热力图")
        fig.tight_layout()
        return fig


class ProfileHelperAPI:
    def __init__(self, *, dataset_id: str | None):
        self._dataset_id = dataset_id

    def schema(self) -> dict[str, Any]:
        dataset_id = self._require_dataset_id()
        return get_schema_profile(dataset_id)

    def analysis_preprocess(self) -> dict[str, Any]:
        dataset_id = self._require_dataset_id()
        return get_analysis_preprocess_artifact(dataset_id)

    def model_prep_plan(self, target: str | None = None, features: list[str] | None = None) -> dict[str, Any]:
        dataset_id = self._require_dataset_id()
        try:
            return get_model_prep_plan(dataset_id, target=target, features=features)
        except DatasetLoadError as exc:
            raise SafeExecutionError(exc.message) from exc

    def latest(self, artifact_type: str | None = None) -> dict[str, Any]:
        dataset_id = self._require_dataset_id()
        artifact = get_artifact_repository().get_latest(dataset_id, artifact_type=artifact_type)
        if artifact is None:
            raise SafeExecutionError("当前还没有可复用的数据理解结果。")
        return artifact

    def _require_dataset_id(self) -> str:
        if not self._dataset_id:
            raise SafeExecutionError("当前没有可用数据集。请先上传数据。")
        return self._dataset_id


class MLHelperAPI:
    def __init__(self, df: pd.DataFrame, *, dataset_id: str | None):
        self._df = df
        self._dataset_id = dataset_id

    def logistic_fit(
        self,
        target: str,
        features: list[str] | None = None,
        test_size: float | None = None,
        positive_label: Any = None,
    ) -> dict[str, Any]:
        service = self._service()
        try:
            return service.logistic_fit(
                target,
                features=features,
                test_size=test_size,
                positive_label=positive_label,
            )
        except DatasetLoadError as exc:
            raise SafeExecutionError(exc.message) from exc
        except MLHelperError as exc:
            raise SafeExecutionError(str(exc)) from exc

    def linear_regression_fit(
        self,
        target: str,
        features: list[str] | None = None,
        test_size: float | None = None,
    ) -> dict[str, Any]:
        service = self._service()
        try:
            return service.linear_regression_fit(
                target,
                features=features,
                test_size=test_size,
            )
        except DatasetLoadError as exc:
            raise SafeExecutionError(exc.message) from exc
        except MLHelperError as exc:
            raise SafeExecutionError(str(exc)) from exc

    def metrics(self, model_artifact_id: str | None = None) -> dict[str, Any]:
        service = self._service()
        try:
            return service.metrics(model_artifact_id=model_artifact_id)
        except MLHelperError as exc:
            raise SafeExecutionError(str(exc)) from exc

    def feature_importance(self, model_artifact_id: str | None = None, top_k: int = 10) -> dict[str, Any]:
        service = self._service()
        try:
            return service.feature_importance(model_artifact_id=model_artifact_id, top_k=top_k)
        except MLHelperError as exc:
            raise SafeExecutionError(str(exc)) from exc

    def latest(self, artifact_type: str | None = None) -> dict[str, Any]:
        service = self._service()
        try:
            return service.latest(artifact_type=artifact_type)
        except MLHelperError as exc:
            raise SafeExecutionError(str(exc)) from exc

    def _service(self) -> BaselineMLService:
        dataset_id = self._dataset_id
        if not dataset_id:
            raise SafeExecutionError("当前没有可用数据集。请先上传 CSV 后再训练模型。")
        return BaselineMLService(dataset_id=dataset_id)


class StatsHelperAPI:
    MAX_AUTO_NUMERIC_COLUMNS = SETTINGS.max_auto_numeric_columns
    MAX_AUTO_CATEGORICAL_COLUMNS = SETTINGS.max_auto_categorical_columns
    MAX_CORR_COLUMNS = SETTINGS.max_corr_columns
    MAX_TOP_PAIRS = SETTINGS.max_top_pairs
    MAX_GROUP_ROWS = SETTINGS.max_group_rows

    def __init__(self, df: pd.DataFrame, *, dataset_id: str | None):
        self._df = df
        self._dataset_id = dataset_id

    def latest(self, artifact_type: str | None = None) -> dict[str, Any]:
        artifact = get_artifact_repository().get_latest(self._dataset_id, artifact_type=artifact_type)
        if artifact is None:
            raise SafeExecutionError("当前还没有可复用的统计结果。请先执行一次统计分析。")
        return artifact

    def describe_numeric(self, columns: list[str] | None = None) -> dict[str, Any]:
        selected_columns, warnings = self._resolve_numeric_columns(columns)
        if not selected_columns:
            raise SafeExecutionError("没有可用于数值描述统计的列。")

        rows: list[dict[str, Any]] = []
        for column in selected_columns:
            series = pd.to_numeric(self._df[column], errors="coerce")
            non_null = series.dropna()
            stats_row = {
                "column": column,
                "count": int(non_null.count()),
                "mean": self._safe_float(non_null.mean()),
                "std": self._safe_float(non_null.std(ddof=1)),
                "min": self._safe_float(non_null.min()),
                "25%": self._safe_float(non_null.quantile(0.25)),
                "50%": self._safe_float(non_null.quantile(0.50)),
                "75%": self._safe_float(non_null.quantile(0.75)),
                "max": self._safe_float(non_null.max()),
                "missing_count": int(series.isna().sum()),
            }
            rows.append(stats_row)

        artifact = build_artifact(
            artifact_type="stats_result",
            dataset_id=self._dataset_id,
            payload={"stats_type": "describe_numeric", "columns": selected_columns, "rows": rows},
            warnings=warnings,
        )
        return get_artifact_repository().register(self._dataset_id, artifact)

    def describe_categorical(self, columns: list[str] | None = None) -> dict[str, Any]:
        selected_columns, warnings = self._resolve_categorical_columns(columns)
        if not selected_columns:
            raise SafeExecutionError("没有可用于分类描述统计的列。")

        rows: list[dict[str, Any]] = []
        for column in selected_columns:
            series = self._df[column]
            non_null = series.dropna()
            top_value = None
            top_freq = 0
            if not non_null.empty:
                vc = non_null.value_counts(dropna=False)
                top_value = self._convert_scalar(vc.index[0])
                top_freq = int(vc.iloc[0])
            rows.append(
                {
                    "column": column,
                    "non_null_count": int(non_null.count()),
                    "missing_count": int(series.isna().sum()),
                    "unique_count": int(non_null.nunique(dropna=True)),
                    "top_value": top_value,
                    "top_freq": top_freq,
                }
            )

        artifact = build_artifact(
            artifact_type="stats_result",
            dataset_id=self._dataset_id,
            payload={"stats_type": "describe_categorical", "columns": selected_columns, "rows": rows},
            warnings=warnings,
        )
        return get_artifact_repository().register(self._dataset_id, artifact)

    def group_summary(
        self,
        group_by: str,
        metrics: list[dict[str, Any]] | None = None,
        sort_by: str | None = None,
        ascending: bool = False,
        top_n: int | None = SETTINGS.default_group_top_n,
    ) -> dict[str, Any]:
        if group_by not in self._df.columns:
            raise SafeExecutionError(f"列不存在: {group_by}")
        if top_n is not None and top_n <= 0:
            raise SafeExecutionError("top_n 必须为正整数。")

        metrics = metrics or [{"op": "count", "as": "row_count"}]
        agg_map: dict[str, tuple[str | None, str, Any]] = {}
        for idx, metric in enumerate(metrics):
            op = str(metric.get("op", "")).lower()
            alias = str(metric.get("as") or f"metric_{idx}")
            column = metric.get("column")
            if op not in {"count", "mean", "median", "sum", "min", "max", "nunique", "rate"}:
                raise SafeExecutionError(f"不支持的聚合操作: {op}")
            if op == "count":
                agg_map[alias] = (None, "count", None)
                continue
            if not isinstance(column, str) or column not in self._df.columns:
                raise SafeExecutionError(f"列不存在: {column}")
            if op in {"mean", "median", "sum", "min", "max"} and not pd.api.types.is_numeric_dtype(self._df[column]):
                raise SafeExecutionError(f"{op} 仅支持数值列: {column}")
            positive_label = metric.get("positive_label") if op == "rate" else None
            agg_map[alias] = (column, op, positive_label)

        grouped = self._df.groupby(group_by, dropna=False)
        result = pd.DataFrame(index=grouped.size().index)
        warnings: list[str] = []
        rate_metadata: list[dict[str, Any]] = []
        for alias, (column, op, positive_label) in agg_map.items():
            if op == "count":
                result[alias] = grouped.size()
            elif op == "rate":
                assert column is not None
                series = self._df[column]
                inference = self._infer_positive_label(series, explicit_label=positive_label)
                if inference["warning"]:
                    warnings.append(str(inference["warning"]))
                if inference["source"] == "ambiguous" or inference["positive_label"] is None:
                    mapped = pd.Series(np.nan, index=series.index, dtype="float64")
                else:
                    mapped = self._map_positive_rate(series, inference["positive_label"])
                temp = self._df.copy(deep=False)
                temp["_rate_value_"] = mapped
                result[alias] = temp.groupby(group_by, dropna=False)["_rate_value_"].mean()
                rate_metadata.append(
                    {
                        "metric": alias,
                        "source_column": column,
                        "positive_label": self._convert_scalar(inference["positive_label"]),
                        "positive_label_source": inference["source"],
                        "positive_label_warning": inference["warning"],
                    }
                )
            else:
                assert column is not None
                if op == "mean":
                    result[alias] = grouped[column].mean()
                elif op == "median":
                    result[alias] = grouped[column].median()
                elif op == "sum":
                    result[alias] = grouped[column].sum()
                elif op == "min":
                    result[alias] = grouped[column].min()
                elif op == "max":
                    result[alias] = grouped[column].max()
                elif op == "nunique":
                    result[alias] = grouped[column].nunique(dropna=True)

        output = result.reset_index().rename(columns={group_by: "group"})
        if sort_by:
            if sort_by not in output.columns:
                raise SafeExecutionError(f"排序列不存在: {sort_by}")
            output = output.sort_values(sort_by, ascending=ascending, kind="stable")
        else:
            output = output.sort_values("group", ascending=True, kind="stable")

        if len(output.index) > self.MAX_GROUP_ROWS:
            warnings.append(f"分组结果超过 {self.MAX_GROUP_ROWS} 行，已截断。")
            output = output.head(self.MAX_GROUP_ROWS)

        if top_n is not None:
            output = output.head(top_n)

        artifact = build_artifact(
            artifact_type="stats_result",
            dataset_id=self._dataset_id,
            payload={
                "stats_type": "group_summary",
                "group_by": group_by,
                "metrics": metrics,
                "rows": self._records(output),
                "row_count": int(len(output.index)),
                "rate_metadata": rate_metadata,
            },
            warnings=warnings,
        )
        return get_artifact_repository().register(self._dataset_id, artifact)

    def correlation(self, columns: list[str] | None = None, top_k: int = 10) -> dict[str, Any]:
        if top_k <= 0:
            raise SafeExecutionError("top_k 必须为正整数。")
        if columns:
            self._validate_columns(columns)
            selected = [col for col in columns if pd.api.types.is_numeric_dtype(self._df[col])]
            warnings = []
            dropped = [col for col in columns if col not in selected]
            if dropped:
                warnings.append(f"以下列非数值类型，已忽略：{', '.join(dropped)}")
        else:
            selected = [str(col) for col in self._df.select_dtypes(include=[np.number]).columns]
            warnings = []
            if len(selected) > self.MAX_CORR_COLUMNS:
                warnings.append(f"自动选择数值列超过 {self.MAX_CORR_COLUMNS}，仅保留前 {self.MAX_CORR_COLUMNS} 列。")
                selected = selected[: self.MAX_CORR_COLUMNS]

        if len(selected) < 2:
            raise SafeExecutionError("相关性分析至少需要两列数值列。")

        corr_df = self._df[selected].corr(method="pearson", numeric_only=True).fillna(0.0)
        top_pairs: list[dict[str, Any]] = []
        for i, col_a in enumerate(selected):
            for col_b in selected[i + 1 :]:
                value = float(corr_df.loc[col_a, col_b])
                top_pairs.append(
                    {
                        "col_a": col_a,
                        "col_b": col_b,
                        "corr": round(value, 6),
                        "abs_corr": round(abs(value), 6),
                    }
                )
        top_pairs = sorted(top_pairs, key=lambda item: item["abs_corr"], reverse=True)[: min(top_k, self.MAX_TOP_PAIRS)]

        artifact = build_artifact(
            artifact_type="stats_result",
            dataset_id=self._dataset_id,
            payload={
                "stats_type": "correlation",
                "columns": selected,
                "matrix": self._records(corr_df.reset_index().rename(columns={"index": "column"})),
                "top_pairs": top_pairs,
            },
            warnings=warnings,
        )
        return get_artifact_repository().register(self._dataset_id, artifact)

    def t_test(self, value_col: str, group_col: str, group_a: Any, group_b: Any) -> dict[str, Any]:
        self._validate_columns([value_col, group_col])
        if not pd.api.types.is_numeric_dtype(self._df[value_col]):
            raise SafeExecutionError(f"t 检验仅支持数值列: {value_col}")

        subset = self._df[[value_col, group_col]].dropna()
        a_values = pd.to_numeric(subset[subset[group_col] == group_a][value_col], errors="coerce").dropna()
        b_values = pd.to_numeric(subset[subset[group_col] == group_b][value_col], errors="coerce").dropna()
        warnings: list[str] = []
        if len(a_values.index) < 2 or len(b_values.index) < 2:
            warnings.append("样本量过小，t 检验结果可能不稳定（每组至少建议 2 条）。")

        if a_values.empty or b_values.empty:
            statistic = None
            p_value = None
        else:
            statistic, p_value = scipy_stats.ttest_ind(a_values, b_values, equal_var=False, nan_policy="omit")

        artifact = build_artifact(
            artifact_type="test_result",
            dataset_id=self._dataset_id,
            payload={
                "test_type": "t_test",
                "value_col": value_col,
                "group_col": group_col,
                "group_a": self._convert_scalar(group_a),
                "group_b": self._convert_scalar(group_b),
                "statistic": self._safe_float(statistic),
                "p_value": self._safe_float(p_value),
                "group_a_mean": self._safe_float(a_values.mean() if not a_values.empty else None),
                "group_b_mean": self._safe_float(b_values.mean() if not b_values.empty else None),
                "group_a_size": int(len(a_values.index)),
                "group_b_size": int(len(b_values.index)),
                "interpretation": self._interpret_p_value(p_value),
            },
            warnings=warnings,
        )
        return get_artifact_repository().register(self._dataset_id, artifact)

    def chi_square(self, col_a: str, col_b: str) -> dict[str, Any]:
        self._validate_columns([col_a, col_b])
        contingency = pd.crosstab(self._df[col_a], self._df[col_b], dropna=False)
        if contingency.empty:
            raise SafeExecutionError("卡方检验失败：分组后没有可用数据。")
        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)
        warnings: list[str] = []
        if (expected < 5).any():
            warnings.append("部分期望频数小于 5，卡方检验前提较弱，请谨慎解释。")

        artifact = build_artifact(
            artifact_type="test_result",
            dataset_id=self._dataset_id,
            payload={
                "test_type": "chi_square",
                "col_a": col_a,
                "col_b": col_b,
                "statistic": self._safe_float(chi2),
                "p_value": self._safe_float(p_value),
                "dof": int(dof),
                "interpretation": self._interpret_p_value(p_value),
                "contingency_rows": self._records(contingency.reset_index()),
            },
            warnings=warnings,
        )
        return get_artifact_repository().register(self._dataset_id, artifact)

    def anova(self, value_col: str, group_col: str) -> dict[str, Any]:
        self._validate_columns([value_col, group_col])
        if not pd.api.types.is_numeric_dtype(self._df[value_col]):
            raise SafeExecutionError(f"ANOVA 仅支持数值列: {value_col}")
        subset = self._df[[value_col, group_col]].dropna()
        grouped = []
        means: list[dict[str, Any]] = []
        warnings: list[str] = []

        for group_name, frame in subset.groupby(group_col, dropna=False):
            values = pd.to_numeric(frame[value_col], errors="coerce").dropna()
            if values.empty:
                continue
            grouped.append(values)
            means.append(
                {
                    "group": self._convert_scalar(group_name),
                    "size": int(len(values.index)),
                    "mean": self._safe_float(values.mean()),
                }
            )
            if len(values.index) < 2:
                warnings.append(f"分组 {group_name} 样本量小于 2，结果可能不稳定。")

        if len(grouped) < 3:
            raise SafeExecutionError("ANOVA 至少需要 3 个有效分组。")

        statistic, p_value = scipy_stats.f_oneway(*grouped)
        artifact = build_artifact(
            artifact_type="test_result",
            dataset_id=self._dataset_id,
            payload={
                "test_type": "anova",
                "value_col": value_col,
                "group_col": group_col,
                "statistic": self._safe_float(statistic),
                "p_value": self._safe_float(p_value),
                "group_stats": means,
                "interpretation": self._interpret_p_value(p_value),
            },
            warnings=warnings,
        )
        return get_artifact_repository().register(self._dataset_id, artifact)

    def _resolve_numeric_columns(self, columns: list[str] | None) -> tuple[list[str], list[str]]:
        warnings: list[str] = []
        if columns:
            self._validate_columns(columns)
            selected = [col for col in columns if pd.api.types.is_numeric_dtype(self._df[col])]
            dropped = [col for col in columns if col not in selected]
            if dropped:
                warnings.append(f"以下列非数值类型，已忽略：{', '.join(dropped)}")
            return selected, warnings

        selected = [str(col) for col in self._df.select_dtypes(include=[np.number]).columns]
        if len(selected) > self.MAX_AUTO_NUMERIC_COLUMNS:
            warnings.append(
                f"自动选择数值列超过 {self.MAX_AUTO_NUMERIC_COLUMNS}，仅保留前 {self.MAX_AUTO_NUMERIC_COLUMNS} 列。"
            )
            selected = selected[: self.MAX_AUTO_NUMERIC_COLUMNS]
        return selected, warnings

    def _resolve_categorical_columns(self, columns: list[str] | None) -> tuple[list[str], list[str]]:
        warnings: list[str] = []
        if columns:
            self._validate_columns(columns)
            selected = [col for col in columns if not pd.api.types.is_numeric_dtype(self._df[col])]
            dropped = [col for col in columns if col not in selected]
            if dropped:
                warnings.append(f"以下列为数值列，已忽略：{', '.join(dropped)}")
            return selected, warnings

        selected = [str(col) for col in self._df.columns if not pd.api.types.is_numeric_dtype(self._df[col])]
        if len(selected) > self.MAX_AUTO_CATEGORICAL_COLUMNS:
            warnings.append(
                f"自动选择分类列超过 {self.MAX_AUTO_CATEGORICAL_COLUMNS}，仅保留前 {self.MAX_AUTO_CATEGORICAL_COLUMNS} 列。"
            )
            selected = selected[: self.MAX_AUTO_CATEGORICAL_COLUMNS]
        return selected, warnings

    def _validate_columns(self, columns: list[str]) -> None:
        for column in columns:
            if column not in self._df.columns:
                raise SafeExecutionError(f"列不存在: {column}")

    def _infer_positive_label(self, series: pd.Series, explicit_label: Any = None) -> dict[str, Any]:
        if explicit_label is not None:
            return {
                "positive_label": explicit_label,
                "source": "explicit",
                "warning": None,
            }

        non_null = series.dropna()
        if non_null.empty:
            return {
                "positive_label": None,
                "source": "ambiguous",
                "warning": "列为空，无法推断正类标签；rate 结果将为空。",
            }

        if pd.api.types.is_bool_dtype(non_null):
            return {"positive_label": True, "source": "boolean_default", "warning": None}

        if pd.api.types.is_numeric_dtype(non_null):
            unique_numeric = sorted(set(pd.to_numeric(non_null, errors="coerce").dropna().tolist()))
            if unique_numeric == [0, 1] or unique_numeric == [0.0, 1.0]:
                return {"positive_label": 1, "source": "binary_numeric_default", "warning": None}
            if len(unique_numeric) == 2:
                return {
                    "positive_label": max(unique_numeric),
                    "source": "binary_numeric_default",
                    "warning": f"列包含二值数值 {unique_numeric}，按较大值作为正类。",
                }
            return {
                "positive_label": None,
                "source": "ambiguous",
                "warning": "rate 仅建议用于二值标签列，当前列无法可靠推断正类。",
            }

        normalized = non_null.astype(str).str.strip()
        unique_text = [item for item in sorted(normalized.unique().tolist()) if item != ""]
        if len(unique_text) != 2:
            return {
                "positive_label": None,
                "source": "ambiguous",
                "warning": f"字符串标签为 {len(unique_text)} 类，无法可靠推断正类；rate 结果将为空。",
            }

        normalized_map = {item.lower(): item for item in unique_text}
        positive_hits = [value for key, value in normalized_map.items() if key in SETTINGS.positive_label_hints]
        negative_hits = [value for key, value in normalized_map.items() if key in SETTINGS.negative_label_hints]

        if len(positive_hits) == 1:
            return {"positive_label": positive_hits[0], "source": "hint_match", "warning": None}

        if len(negative_hits) == 1:
            for item in unique_text:
                if item != negative_hits[0]:
                    return {
                        "positive_label": item,
                        "source": "hint_match",
                        "warning": f"按负类标签 {negative_hits[0]} 反推正类 {item}。",
                    }

        return {
            "positive_label": None,
            "source": "ambiguous",
            "warning": f"未能在标签 {unique_text} 中识别稳定正类，请显式提供 positive_label。",
        }

    def _map_positive_rate(self, series: pd.Series, positive_label: Any) -> pd.Series:
        if isinstance(positive_label, bool):
            return series.fillna(False).astype(bool).eq(bool(positive_label)).astype(int)
        if isinstance(positive_label, (int, float, np.integer, np.floating)):
            numeric_series = pd.to_numeric(series, errors="coerce")
            return numeric_series.eq(float(positive_label)).astype(int)

        normalized = series.astype(str).str.strip().str.lower()
        positive = str(positive_label).strip().lower()
        return normalized.eq(positive).astype(int)

    def _records(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        return frame.replace({np.nan: None}).to_dict(orient="records")

    def _convert_scalar(self, value: Any) -> Any:
        if isinstance(value, (np.integer, np.int64)):
            return int(value)
        if isinstance(value, (np.floating, np.float64)):
            return float(value)
        return value

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            if pd.isna(value):
                return None
            return round(float(value), 6)
        return None

    def _interpret_p_value(self, p_value: Any) -> str:
        p = self._safe_float(p_value)
        if p is None:
            return "样本不足，无法稳定判断统计显著性。"
        if p < 0.01:
            return "差异/关联显著（p < 0.01）。"
        if p < 0.05:
            return "差异/关联较显著（p < 0.05）。"
        return "未观察到显著差异/关联（p >= 0.05）。"


class SafePythonExecutor:
    # This is a constrained execution layer, not an OS-level sandbox.
    # Keep the policy strict and prefer helper APIs over broad Python objects.
    def __init__(self, *, image_dir: Path):
        self.image_dir = image_dir

    def safe_execute_python(self, py_code: str, env: dict[str, Any], *, df: pd.DataFrame | None = None) -> str:
        compiled = self._validate_and_compile(py_code)
        output = []
        execution_env = self._build_env(env)
        timeout_seconds = self._resolve_timeout_seconds(df=df, mode="python")

        try:
            with contextlib.redirect_stdout(_StdoutCollector(output)):
                with _execution_timeout(timeout_seconds):
                    exec(compiled, execution_env, execution_env)
        except SafeExecutionError:
            raise
        except Exception as exc:
            logger.warning("Safe Python execution failed: %s", exc)
            return f"代码执行失败: {exc}"

        printed = "".join(output).strip()
        if printed:
            return printed
        return "代码执行成功，但没有输出。请使用 print() 展示结果。"

    def safe_execute_plot(
        self,
        py_code: str,
        env: dict[str, Any],
        figure_name: str | None = None,
        *,
        df: pd.DataFrame | None = None,
    ) -> str:
        compiled = self._validate_and_compile(py_code)
        execution_env = self._build_env(env)
        timeout_seconds = self._resolve_timeout_seconds(df=df, mode="plot")

        plt.close("all")

        try:
            with _execution_timeout(timeout_seconds):
                exec(compiled, execution_env, execution_env)
        except SafeExecutionError:
            raise
        except Exception as exc:
            logger.warning("Safe plot execution failed: %s", exc)
            return f"绘图执行失败: {exc}"

        figure = execution_env.get(figure_name) if figure_name else None
        if figure is None:
            figure = plt.gcf()

        if figure is None:
            return "绘图代码执行完毕，但未生成图像对象。"

        if hasattr(figure, "figure") and not hasattr(figure, "savefig"):
            figure = getattr(figure, "figure", figure)

        filename = f"{uuid4().hex}.png"
        save_path = (self.image_dir / filename).resolve()
        if save_path.parent != self.image_dir:
            raise SafeExecutionError("图片保存路径非法。")
        figure.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close("all")

        dataset_id = get_current_dataset_id()
        if dataset_id:
            register_dataset_generated_image(dataset_id, filename)

        set_current_image_event(
            {
                "type": "image_generated",
                "filename": filename,
                "tool_name": "fig_inter",
            }
        )
        return "图表已生成"

    def _validate_and_compile(self, py_code: str):
        try:
            tree = ast.parse(py_code, mode="exec")
        except SyntaxError as exc:
            raise SafeExecutionError(f"Python 语法错误: {exc.msg}") from exc

        SafeCodeValidator().visit(tree)
        return compile(tree, "<safe-python>", "exec")

    def _build_env(self, env: dict[str, Any]) -> dict[str, Any]:
        safe_env = {"__builtins__": ALLOWED_BUILTINS.copy()}
        safe_env.update(env)
        return safe_env

    def _resolve_timeout_seconds(self, *, df: pd.DataFrame | None, mode: Literal["python", "plot"]) -> float:
        row_count = 0 if df is None else int(len(df.index))
        column_count = 0 if df is None else int(len(df.columns))
        complexity_bonus = min(4.0, row_count / 4000.0 + column_count / 25.0)
        if mode == "plot":
            base = SETTINGS.plot_execution_timeout_base_seconds
            ceiling = SETTINGS.plot_execution_timeout_max_seconds
        else:
            base = SETTINGS.python_execution_timeout_base_seconds
            ceiling = SETTINGS.python_execution_timeout_max_seconds
        return min(ceiling, base + complexity_bonus)


class ReadOnlyDataFrameProxy:
    """
    Narrow dataframe view exposed to LLM-generated code.
    It intentionally blocks private attributes and write/network/file paths.
    """

    _ALLOWED_METHODS = {
        "copy",
        "corr",
        "describe",
        "dropna",
        "groupby",
        "head",
        "isna",
        "mean",
        "median",
        "nunique",
        "quantile",
        "sample",
        "select_dtypes",
        "sort_values",
        "sum",
        "tail",
        "value_counts",
    }

    def __init__(self, df: pd.DataFrame):
        self._source = df.copy(deep=True)

    @property
    def columns(self) -> pd.Index:
        return self._source.columns.copy()

    @property
    def dtypes(self) -> pd.Series:
        return self._source.dtypes.copy()

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._source.shape)

    def __len__(self) -> int:
        return int(len(self._source.index))

    def __getitem__(self, key: Any) -> Any:
        key = _unwrap_read_only_pandas_proxy(key)
        value = self._source.__getitem__(key)
        if isinstance(value, pd.DataFrame):
            return ReadOnlyDataFrameProxy(value)
        if isinstance(value, pd.Series):
            return ReadOnlySeriesProxy(value)
        return value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise SafeExecutionError(f"不允许访问敏感属性: {name}")
        if name in FORBIDDEN_METHOD_NAMES:
            raise SafeExecutionError(f"不允许调用危险函数: {name}")
        if name not in self._ALLOWED_METHODS:
            raise SafeExecutionError(f"df 当前仅支持只读访问，属性不可用: {name}")

        attr = getattr(self._source, name)
        if not callable(attr):
            return attr

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                return ReadOnlyDataFrameProxy(result)
            if isinstance(result, pd.Series):
                return ReadOnlySeriesProxy(result)
            return result

        return _wrapped


class ReadOnlySeriesProxy:
    """Read-only Series view returned from dataframe indexing."""

    _ALLOWED_METHODS = {
        "corr",
        "describe",
        "drop_duplicates",
        "dropna",
        "head",
        "isna",
        "mean",
        "median",
        "nunique",
        "quantile",
        "sort_values",
        "sum",
        "tail",
        "tolist",
        "unique",
        "value_counts",
    }

    def __init__(self, series: pd.Series):
        self._source = series.copy(deep=True)

    @property
    def name(self) -> Any:
        return self._source.name

    @property
    def dtype(self) -> Any:
        return self._source.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._source.shape)

    def __len__(self) -> int:
        return int(len(self._source.index))

    def __getitem__(self, key: Any) -> Any:
        key = _unwrap_read_only_pandas_proxy(key)
        value = self._source.__getitem__(key)
        if isinstance(value, pd.Series):
            return ReadOnlySeriesProxy(value)
        return value

    def __iter__(self):
        return iter(self._source.copy(deep=True))

    def __eq__(self, other: Any) -> pd.Series:  # type: ignore[override]
        return self._source.eq(_unwrap_read_only_pandas_proxy(other))

    def __ne__(self, other: Any) -> pd.Series:  # type: ignore[override]
        return self._source.ne(_unwrap_read_only_pandas_proxy(other))

    def __lt__(self, other: Any) -> pd.Series:
        return self._source.lt(_unwrap_read_only_pandas_proxy(other))

    def __le__(self, other: Any) -> pd.Series:
        return self._source.le(_unwrap_read_only_pandas_proxy(other))

    def __gt__(self, other: Any) -> pd.Series:
        return self._source.gt(_unwrap_read_only_pandas_proxy(other))

    def __ge__(self, other: Any) -> pd.Series:
        return self._source.ge(_unwrap_read_only_pandas_proxy(other))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise SafeExecutionError(f"不允许访问敏感属性: {name}")
        if name in FORBIDDEN_METHOD_NAMES:
            raise SafeExecutionError(f"不允许调用危险函数: {name}")
        if name not in self._ALLOWED_METHODS:
            raise SafeExecutionError(f"Series 当前仅支持只读访问，属性不可用: {name}")

        attr = getattr(self._source, name)
        if not callable(attr):
            return attr

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                return ReadOnlyDataFrameProxy(result)
            if isinstance(result, pd.Series):
                return ReadOnlySeriesProxy(result)
            return result

        return _wrapped


def _unwrap_read_only_pandas_proxy(value: Any) -> Any:
    if isinstance(value, ReadOnlyDataFrameProxy):
        return value._source.copy(deep=True)
    if isinstance(value, ReadOnlySeriesProxy):
        return value._source.copy(deep=True)
    return value


class _StdoutCollector:
    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    def write(self, value: str) -> int:
        self._chunks.append(value)
        return len(value)

    def flush(self) -> None:  # pragma: no cover - interface only
        return None


@contextlib.contextmanager
def _execution_timeout(seconds: float):
    deadline = time.monotonic() + seconds
    previous_trace = sys.gettrace()

    def _trace(frame: Any, event: str, arg: Any):  # pragma: no cover - runtime guard
        if event == "line" and time.monotonic() > deadline:
            raise ToolExecutionTimeoutError(seconds)
        return _trace

    sys.settrace(_trace)
    try:
        yield
    finally:
        sys.settrace(previous_trace)


def set_current_dataset_id(dataset_id: str | None) -> None:
    CURRENT_DATASET_ID.set(dataset_id)


def get_current_dataset_id() -> str | None:
    return CURRENT_DATASET_ID.get()


@contextlib.contextmanager
def bind_current_dataset_id(dataset_id: str | None):
    token: Token[str | None] = CURRENT_DATASET_ID.set(dataset_id)
    try:
        yield dataset_id
    finally:
        CURRENT_DATASET_ID.reset(token)


def consume_current_image_event() -> dict[str, Any] | None:
    event = CURRENT_IMAGE_EVENT_STATE.get()
    CURRENT_IMAGE_EVENT_STATE.set(None)
    return event


def set_current_image_event(event: dict[str, Any] | None) -> None:
    CURRENT_IMAGE_EVENT_STATE.set(event)


def configure_fonts() -> None:
    system_name = platform.system()
    if system_name == "Windows":
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    elif system_name == "Darwin":
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC"]
    else:
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


configure_fonts()

BACKEND_ROOT = Path(__file__).resolve().parents[1]
STATIC_IMAGES_DIR = (BACKEND_ROOT / "static" / "images").resolve()
STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
EXECUTOR = SafePythonExecutor(image_dir=STATIC_IMAGES_DIR)


class PythonCodeInput(BaseModel):
    py_code: str = Field(description="Python代码。可以使用变量 df、data、viz、stats、profile、ml。")


class FigCodeInput(BaseModel):
    py_code: str = Field(description="绘图代码。需生成图像对象。")
    fname: str = Field(description="图像变量名，例如 'fig'。")


class MLLogisticFitInput(BaseModel):
    target: str = Field(description="二分类目标列名称，例如 Churn。")
    features: list[str] | None = Field(default=None, description="可选特征列名单。")
    test_size: float | None = Field(default=None, description="可选测试集比例。")
    positive_label: Any | None = Field(default=None, description="可选正类标签。")


class MLLinearRegressionFitInput(BaseModel):
    target: str = Field(description="数值回归目标列名称。")
    features: list[str] | None = Field(default=None, description="可选特征列名单。")
    test_size: float | None = Field(default=None, description="可选测试集比例。")


class MLMetricsInput(BaseModel):
    model_artifact_id: str | None = Field(default=None, description="可选模型 artifact ID。")


class MLFeatureImportanceInput(BaseModel):
    model_artifact_id: str | None = Field(default=None, description="可选模型 artifact ID。")
    top_k: int = Field(default=10, description="返回前几个重要特征。")


class MLLatestInput(BaseModel):
    artifact_type: str | None = Field(default=None, description="可选 artifact 类型，例如 model_result。")


class MLExecuteInput(BaseModel):
    action: Literal["train", "metrics", "feature_importance", "latest"] = Field(
        description="要执行的 ML 动作。"
    )
    model_type: Literal["logistic_regression", "linear_regression"] | None = Field(
        default=None,
        description="训练动作时使用的模型类型。",
    )
    target: str | None = Field(default=None, description="训练动作的目标列。")
    features: list[str] | None = Field(default=None, description="可选特征列名单。")
    test_size: float | None = Field(default=None, description="可选测试集比例。")
    positive_label: Any | None = Field(default=None, description="可选正类标签。")
    model_artifact_id: str | None = Field(default=None, description="可选模型 artifact ID。")
    top_k: int = Field(default=10, description="返回前几个重要特征。")
    artifact_type: str | None = Field(default=None, description="查询 latest 时的 artifact 类型。")


class StatsExecuteInput(BaseModel):
    action: Literal["describe_numeric", "describe_categorical", "group_summary", "correlation", "t_test", "chi_square", "anova", "latest"] = Field(
        description="要执行的统计动作。"
    )
    columns: list[str] | None = Field(default=None, description="描述统计或相关性分析所需列。")
    group_by: str | None = Field(default=None, description="分组汇总使用的分组列。")
    metrics: list[dict[str, Any]] | None = Field(default=None, description="分组汇总的指标定义。")
    sort_by: str | None = Field(default=None, description="分组汇总的排序列。")
    ascending: bool = Field(default=False, description="是否升序排序。")
    top_n: int | None = Field(default=None, description="分组汇总返回前几行。")
    value_col: str | None = Field(default=None, description="t 检验或 ANOVA 的数值列。")
    group_col: str | None = Field(default=None, description="t 检验、ANOVA 或卡方检验的分组列。")
    group_a: Any | None = Field(default=None, description="t 检验组 A 的标签。")
    group_b: Any | None = Field(default=None, description="t 检验组 B 的标签。")
    col_a: str | None = Field(default=None, description="卡方检验的列 A。")
    col_b: str | None = Field(default=None, description="卡方检验的列 B。")
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


def _build_helper_api(df: pd.DataFrame) -> tuple[DataHelperAPI, PlotHelperAPI, StatsHelperAPI, ProfileHelperAPI, MLHelperAPI]:
    dataset_id = get_current_dataset_id()
    return (
        DataHelperAPI(df),
        PlotHelperAPI(df),
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
    return result.startswith("代码执行失败:") or result.startswith("绘图执行失败:")


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
    安全执行受限的数据分析代码，仅暴露白名单 helper API。
    """
    start = time.perf_counter()
    dataset_id = get_current_dataset_id()
    df = _get_dataset_df()
    if df is None:
        result = "错误：当前没有可用数据集。请先上传数据。"
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
        data, viz, stats, profile, ml = _build_helper_api(df)
        env = {
            "df": ReadOnlyDataFrameProxy(df),
            "data": data,
            "viz": viz,
            "stats": stats,
            "profile": profile,
            "ml": ml,
        }
        result = EXECUTOR.safe_execute_python(py_code, env, df=df)
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
        result = f"代码被安全策略拦截: {exc}"
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


@tool(args_schema=FigCodeInput)
def fig_inter(py_code: str, fname: str) -> str:
    """
    安全执行绘图代码并保存生成的图片。
    """
    start = time.perf_counter()
    dataset_id = get_current_dataset_id()
    tool_args = _audit_tool_args(fname=fname)
    df = _get_dataset_df()
    if df is None:
        result = "错误：当前没有可用数据集。请先上传数据。"
        _record_tool_audit(
            tool_name="fig_inter",
            dataset_id=dataset_id,
            tool_args=tool_args,
            code=py_code,
            execution_status="success",
            start=start,
            result=result,
        )
        return result

    available_columns = _available_columns_for_self_correction(df)
    try:
        data, viz, stats, profile, ml = _build_helper_api(df)
        env = {
            "df": ReadOnlyDataFrameProxy(df),
            "data": data,
            "viz": viz,
            "stats": stats,
            "profile": profile,
            "ml": ml,
        }
        set_current_image_event(None)
        result = EXECUTOR.safe_execute_plot(py_code, env, fname, df=df)
        if _is_execution_failure_text(result):
            _record_tool_audit(
                tool_name="fig_inter",
                dataset_id=dataset_id,
                tool_args=tool_args,
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
                    tool_name="fig_inter",
                ),
            )
            return result
        _record_tool_audit(
            tool_name="fig_inter",
            dataset_id=dataset_id,
            tool_args=tool_args,
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
            tool_name="fig_inter",
        )
        _record_tool_audit(
            tool_name="fig_inter",
            dataset_id=dataset_id,
            tool_args=tool_args,
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
                tool_name="fig_inter",
            ),
        )
        return result
    except SafeExecutionError as exc:
        result = f"绘图代码被安全策略拦截: {exc}"
        _record_tool_audit(
            tool_name="fig_inter",
            dataset_id=dataset_id,
            tool_args=tool_args,
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
                tool_name="fig_inter",
            ),
        )
        return result
    except Exception as exc:
        _record_tool_audit(
            tool_name="fig_inter",
            dataset_id=dataset_id,
            tool_args=tool_args,
            code=py_code,
            execution_status="error",
            start=start,
            error_message=str(exc),
            extra=_structured_error_extra(
                exc,
                dataset_id=dataset_id,
                available_columns=available_columns,
                original_code=py_code,
                tool_name="fig_inter",
            ),
        )
        raise
    finally:
        plt.close("all")


@tool(args_schema=MLLogisticFitInput)
def ml_logistic_fit(target: str, features: list[str] | None = None, test_size: float | None = None, positive_label: Any | None = None) -> str:
    """
    直接训练 baseline 逻辑回归模型，并返回结构化 model_result。
    """
    df = _get_dataset_df()
    if df is None:
        return "错误：当前没有可用数据集。请先上传数据。"

    _, _, _, _, ml = _build_helper_api(df)
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
    直接训练 baseline 线性回归模型，并返回结构化 model_result。
    """
    df = _get_dataset_df()
    if df is None:
        return "错误：当前没有可用数据集。请先上传数据。"

    _, _, _, _, ml = _build_helper_api(df)
    try:
        artifact = ml.linear_regression_fit(target=target, features=features, test_size=test_size)
        return _serialize_tool_payload(artifact)
    except SafeExecutionError as exc:
        return f"错误：{exc}"


@tool(args_schema=MLMetricsInput)
def ml_metrics(model_artifact_id: str | None = None) -> str:
    """
    返回已有模型的 metrics_result。
    """
    df = _get_dataset_df()
    if df is None:
        return "错误：当前没有可用数据集。请先上传数据。"

    _, _, _, _, ml = _build_helper_api(df)
    try:
        artifact = ml.metrics(model_artifact_id=model_artifact_id)
        return _serialize_tool_payload(artifact)
    except SafeExecutionError as exc:
        return f"错误：{exc}"


@tool(args_schema=MLFeatureImportanceInput)
def ml_feature_importance(model_artifact_id: str | None = None, top_k: int = 10) -> str:
    """
    返回已有模型的 feature_importance_result。
    """
    df = _get_dataset_df()
    if df is None:
        return "错误：当前没有可用数据集。请先上传数据。"

    _, _, _, _, ml = _build_helper_api(df)
    try:
        artifact = ml.feature_importance(model_artifact_id=model_artifact_id, top_k=top_k)
        return _serialize_tool_payload(artifact)
    except SafeExecutionError as exc:
        return f"错误：{exc}"


@tool(args_schema=MLLatestInput)
def ml_latest(artifact_type: str | None = None) -> str:
    """
    返回最近一次 ML 结构化结果。
    """
    df = _get_dataset_df()
    if df is None:
        return "错误：当前没有可用数据集。请先上传数据。"

    _, _, _, _, ml = _build_helper_api(df)
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
    ML 请求请优先使用这个工具，不要先走 python_inter。
    - action="train": 训练逻辑回归或线性回归
    - action="metrics": 返回模型指标
    - action="feature_importance": 返回特征重要性
    - action="latest": 返回最近一次 ML 结构化结果
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
        result = "错误：当前没有可用数据集。请先上传数据。"
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
        _, _, _, _, ml = _build_helper_api(df)
        if action == "train":
            if not target:
                result = "错误：训练动作必须提供 target。"
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
    统一的统计分析入口。统计类请求优先使用这个工具，而不是自己写 Python 代码。
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
        result = "错误：当前没有可用数据集。请先上传数据。"
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
                result = "错误：correlation 需要至少提供两列。"
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
