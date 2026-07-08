"""
Statistical analysis service — StatsHelperAPI and DataHelperAPI.
Extracted from tools.py for modularity.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.result_types import build_artifact, get_artifact_repository
from src.safe_executor import SafeExecutionError

logger = logging.getLogger(__name__)


class DataHelperAPI:
    """Lightweight data exploration API exposed to the LLM's Python execution context."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self._df.columns]

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
        ndf = self._df.select_dtypes(include=[np.number])
        return ndf.describe().T if not ndf.empty else pd.DataFrame()

    def missing_summary(self) -> pd.DataFrame:
        total = max(len(self._df.index), 1)
        mc = self._df.isna().sum()
        mr = (mc / total).round(4)
        return pd.DataFrame({"column": mc.index.astype(str), "missing_count": mc.values, "missing_rate": mr.values})

    def value_counts(self, column: str, top_n: int = 10) -> pd.DataFrame:
        if column not in self._df.columns:
            raise SafeExecutionError(f"Column does not exist: {column}")
        return self._df[column].value_counts(dropna=False).head(top_n).reset_index().rename(
            columns={"index": column, column: "count"}
        )

    def unique(self, column: str) -> list[Any]:
        if column not in self._df.columns:
            raise SafeExecutionError(f"Column does not exist: {column}")
        return list(self._df[column].drop_duplicates().tolist())

    def select(self, columns: list[str]) -> pd.DataFrame:
        return self._df.loc[:, columns]

    def filter_equals(self, column: str, value: Any) -> pd.DataFrame:
        if column not in self._df.columns:
            raise SafeExecutionError(f"Column does not exist: {column}")
        return self._df[self._df[column] == value]

    def top_rows(self, column: str, n: int = 5, ascending: bool = False) -> pd.DataFrame:
        if column not in self._df.columns:
            raise SafeExecutionError(f"Column does not exist: {column}")
        return self._df.sort_values(column, ascending=ascending).head(n)

    def group_mean(self, group_column: str, value_column: str) -> pd.DataFrame:
        if group_column not in self._df.columns or value_column not in self._df.columns:
            raise SafeExecutionError(f"Column does not exist: {group_column} / {value_column}")
        return self._df.groupby(group_column, dropna=False)[value_column].mean().reset_index(name=f"{value_column}_mean")

    def group_sum(self, group_column: str, value_column: str) -> pd.DataFrame:
        if group_column not in self._df.columns or value_column not in self._df.columns:
            raise SafeExecutionError(f"Column does not exist: {group_column} / {value_column}")
        return self._df.groupby(group_column, dropna=False)[value_column].sum().reset_index(name=f"{value_column}_sum")

    def correlation(self, col1: str, col2: str) -> float:
        if col1 not in self._df.columns or col2 not in self._df.columns:
            raise SafeExecutionError(f"Column does not exist: {col1} / {col2}")
        v = self._df[col1].corr(self._df[col2])
        return 0.0 if pd.isna(v) else float(v)


class StatsHelperAPI:
    """Comprehensive statistical analysis API exposed to the LLM."""

    def __init__(self, df: pd.DataFrame, *, dataset_id: str | None):
        self._df = df
        self._dataset_id = dataset_id
        self.MAX_AUTO_NUMERIC_COLUMNS = 20
        self.MAX_AUTO_CATEGORICAL_COLUMNS = 20
        self.MAX_CORR_COLUMNS = 12
        self.MAX_TOP_PAIRS = 20
        self.MAX_GROUP_ROWS = 500
        self.DEFAULT_GROUP_TOP_N = 10
        from src.settings import SETTINGS as _s
        self.MAX_AUTO_NUMERIC_COLUMNS = _s.max_auto_numeric_columns
        self.MAX_AUTO_CATEGORICAL_COLUMNS = _s.max_auto_categorical_columns
        self.MAX_CORR_COLUMNS = _s.max_corr_columns
        self.MAX_TOP_PAIRS = _s.max_top_pairs
        self.MAX_GROUP_ROWS = _s.max_group_rows
        self.DEFAULT_GROUP_TOP_N = _s.default_group_top_n

    def _validate_columns(self, columns: list[str]) -> None:
        for col in columns:
            if col not in self._df.columns:
                raise SafeExecutionError(f"Column does not exist: {col}")

    def _convert_scalar(self, value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    def _records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        return df.replace({np.nan: None}).to_dict(orient="records")

    def describe_numeric(self, columns: list[str] | None = None) -> dict[str, Any]:
        if columns:
            self._validate_columns(columns)
            selected = [c for c in columns if pd.api.types.is_numeric_dtype(self._df[c])]
            dropped = [c for c in columns if c not in selected]
        else:
            selected = [str(c) for c in self._df.select_dtypes(include=[np.number]).columns]
            dropped = []
            if len(selected) > self.MAX_AUTO_NUMERIC_COLUMNS:
                selected = selected[: self.MAX_AUTO_NUMERIC_COLUMNS]
        warnings = []
        if dropped:
            warnings.append(f"Non-numeric columns excluded: {', '.join(dropped)}")
        if not selected:
            raise SafeExecutionError("No numeric columns available for descriptive statistics.")
        desc = self._df[selected].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
        result = desc.reset_index().rename(columns={"index": "column"}).replace({np.nan: None}).to_dict(orient="records")
        artifact = build_artifact(artifact_type="stats_result", dataset_id=self._dataset_id, payload={
            "stats_type": "describe_numeric", "columns": selected, "results": result, "row_count": len(result),
        }, warnings=warnings)
        return get_artifact_repository().register(self._dataset_id, artifact)

    def describe_categorical(self, columns: list[str] | None = None) -> dict[str, Any]:
        if columns:
            self._validate_columns(columns)
            selected = [c for c in columns if not pd.api.types.is_numeric_dtype(self._df[c])]
            dropped = [c for c in columns if c not in selected]
        else:
            selected = [str(c) for c in self._df.columns if not pd.api.types.is_numeric_dtype(self._df[c])]
            dropped = []
            if len(selected) > self.MAX_AUTO_CATEGORICAL_COLUMNS:
                selected = selected[: self.MAX_AUTO_CATEGORICAL_COLUMNS]
        warnings = []
        if dropped:
            warnings.append(f"Numeric columns excluded: {', '.join(dropped)}")
        if not selected:
            raise SafeExecutionError("No categorical columns available for descriptive statistics.")
        rows = []
        for col in selected:
            vc = self._df[col].value_counts(dropna=False)
            rows.append({"column": col, "unique_count": int(self._df[col].nunique()), "top_value": str(vc.index[0]) if len(vc.index) > 0 else None, "top_freq": int(vc.iloc[0]) if len(vc.index) > 0 else 0, "missing_count": int(self._df[col].isna().sum())})
        artifact = build_artifact(artifact_type="stats_result", dataset_id=self._dataset_id, payload={
            "stats_type": "describe_categorical", "columns": selected, "results": rows, "row_count": len(rows),
        }, warnings=warnings)
        return get_artifact_repository().register(self._dataset_id, artifact)

    def group_summary(self, group_by: str, metrics: list[dict[str, Any]], sort_by: str | None = None, ascending: bool = False, top_n: int | None = None) -> dict[str, Any]:
        if group_by not in self._df.columns:
            raise SafeExecutionError(f"Column does not exist: {group_by}")
        for m in metrics:
            col = m.get("column")
            if col and col not in self._df.columns:
                raise SafeExecutionError(f"Column does not exist: {col}")
        from src.settings import SETTINGS as _s
        positive_label_hints = _s.positive_label_hints
        negative_label_hints = _s.negative_label_hints

        result = pd.DataFrame()
        rate_metadata = []
        grouped = self._df.groupby(group_by, dropna=False)
        warnings = []

        for m in metrics:
            alias = m.get("alias") or m.get("column", "value")
            op = m.get("op", "mean")
            column = m.get("column")
            if op == "rate":
                if column is None or column not in self._df.columns:
                    raise SafeExecutionError(f"Column does not exist: {column}")
                raw = self._df[[group_by, column]].copy()
                raw["_rate_value_"] = np.nan
                inference = self._infer_positive_label(raw[column], positive_label_hints, negative_label_hints)
                mapped = self._map_to_binary(raw[column], inference.get("positive_label"))
                raw["_rate_value_"] = mapped
                result[alias] = raw.groupby(group_by, dropna=False)["_rate_value_"].mean()
                rate_metadata.append({"metric": alias, "source_column": column, "positive_label": self._convert_scalar(inference.get("positive_label")), "positive_label_source": inference.get("source"), "positive_label_warning": inference.get("warning")})
            else:
                if column is None or column not in self._df.columns:
                    raise SafeExecutionError(f"Column does not exist: {column}")
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
                raise SafeExecutionError(f"Sort column does not exist: {sort_by}")
            output = output.sort_values(sort_by, ascending=ascending, kind="stable")
        else:
            output = output.sort_values("group", ascending=True, kind="stable")
        if len(output.index) > self.MAX_GROUP_ROWS:
            warnings.append(f"Group result exceeds {self.MAX_GROUP_ROWS} rows, truncated.")
            output = output.head(self.MAX_GROUP_ROWS)
        if top_n is not None:
            output = output.head(top_n)
        artifact = build_artifact(artifact_type="stats_result", dataset_id=self._dataset_id, payload={
            "stats_type": "group_summary", "group_by": group_by, "metrics": metrics,
            "rows": self._records(output), "row_count": int(len(output.index)), "rate_metadata": rate_metadata,
        }, warnings=warnings)
        return get_artifact_repository().register(self._dataset_id, artifact)

    def correlation(self, columns: list[str] | None = None, top_k: int = 10) -> dict[str, Any]:
        if top_k <= 0:
            raise SafeExecutionError("top_k must be a positive integer.")
        if columns:
            self._validate_columns(columns)
            selected = [c for c in columns if pd.api.types.is_numeric_dtype(self._df[c])]
            warnings = []
            dropped = [c for c in columns if c not in selected]
            if dropped:
                warnings.append(f"The following columns are non-numeric and were ignored: {', '.join(dropped)}")
        else:
            selected = [str(c) for c in self._df.select_dtypes(include=[np.number]).columns]
            warnings = []
            if len(selected) > self.MAX_CORR_COLUMNS:
                warnings.append(f"Auto-selected numeric columns exceeded {self.MAX_CORR_COLUMNS}, keeping only top {self.MAX_CORR_COLUMNS}.")
                selected = selected[: self.MAX_CORR_COLUMNS]
        if len(selected) < 2:
            raise SafeExecutionError("Correlation analysis requires at least two numeric columns.")
        corr_df = self._df[selected].corr(method="pearson", numeric_only=True).fillna(0.0)
        top_pairs = []
        for i, a in enumerate(selected):
            for b in selected[i + 1 :]:
                v = float(corr_df.loc[a, b])
                top_pairs.append({"col_a": a, "col_b": b, "corr": round(v, 6), "abs_corr": round(abs(v), 6)})
        top_pairs = sorted(top_pairs, key=lambda x: x["abs_corr"], reverse=True)[: min(top_k, self.MAX_TOP_PAIRS)]
        artifact = build_artifact(artifact_type="stats_result", dataset_id=self._dataset_id, payload={
            "stats_type": "correlation", "columns": selected,
            "matrix": self._records(corr_df.reset_index().rename(columns={"index": "column"})),
            "top_pairs": top_pairs,
        }, warnings=warnings)
        return get_artifact_repository().register(self._dataset_id, artifact)

    def t_test(self, value_col: str, group_col: str, group_a: Any, group_b: Any) -> dict[str, Any]:
        self._validate_columns([value_col, group_col])
        if not pd.api.types.is_numeric_dtype(self._df[value_col]):
            raise SafeExecutionError(f"t-test only supports numeric columns: {value_col}")
        subset = self._df[[value_col, group_col]].dropna()
        a = pd.to_numeric(subset[subset[group_col] == group_a][value_col], errors="coerce").dropna()
        b = pd.to_numeric(subset[subset[group_col] == group_b][value_col], errors="coerce").dropna()
        warnings = []
        if len(a.index) < 2 or len(b.index) < 2:
            warnings.append("Sample size too small, t-test results may be unstable (recommended at least 2 per group).")
        if a.empty or b.empty:
            s, p = None, None
        else:
            s, p = scipy_stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        artifact = build_artifact(artifact_type="test_result", dataset_id=self._dataset_id, payload={
            "test_type": "t_test", "value_column": value_col, "group_column": group_col,
            "group_a": str(group_a), "group_b": str(group_b),
            "statistic": self._convert_scalar(s) if s is not None else None,
            "p_value": self._convert_scalar(p) if p is not None else None,
            "count_a": int(len(a.index)), "count_b": int(len(b.index)),
        }, warnings=warnings)
        return get_artifact_repository().register(self._dataset_id, artifact)

    # _infer_positive_label and _map_to_binary kept as private helpers
    def _infer_positive_label(self, series: pd.Series, pos_hints: tuple[str, ...], neg_hints: tuple[str, ...]) -> dict[str, Any]:
        unique = series.dropna().unique()
        if len(unique) == 1:
            return {"positive_label": self._convert_scalar(unique[0]), "source": "single_value", "warning": None}
        unique_str = sorted(str(u).strip().lower() for u in unique if pd.notna(u))
        for hint in pos_hints:
            if hint in unique_str:
                original = [u for u in unique if str(u).strip().lower() == hint]
                if original:
                    return {"positive_label": self._convert_scalar(original[0]), "source": "positive_hint", "warning": None}
        for hint in neg_hints:
            if hint in unique_str:
                remaining = [u for u in unique if str(u).strip().lower() != hint]
                if len(remaining) == 1:
                    return {"positive_label": self._convert_scalar(remaining[0]), "source": "negative_hint_inverse", "warning": None}
        return {"positive_label": None, "source": "ambiguous", "warning": "Could not reliably infer positive class. Please provide positive_label explicitly."}

    def _map_to_binary(self, series: pd.Series, positive_label: Any) -> pd.Series:
        if positive_label is None:
            return pd.Series([np.nan] * len(series), index=series.index)
        return series.apply(lambda x: 1 if x == positive_label else 0)
