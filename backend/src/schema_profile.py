from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.settings import SETTINGS

IDENTIFIER_NAME_HINTS = ("id", "uuid", "key", "code", "number", "编号", "账号")
TEXT_NAME_HINTS = ("note", "comment", "remark", "desc", "text", "说明", "备注")
TARGET_NAME_HINTS = ("target", "label", "churn", "outcome", "是否")


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    total_rows = int(len(df.index))
    rows_for_ratio = max(total_rows, 1)
    columns: list[dict[str, Any]] = []
    warnings: list[str] = []

    for column in df.columns:
        series = df[column]
        non_null_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = round(unique_count / max(non_null_count, 1), 6) if non_null_count else 0.0
        missing_ratio = round(missing_count / rows_for_ratio, 6)
        sample_values = _sample_values(series)
        notes: list[str] = []

        semantic_type = _infer_semantic_type(
            column_name=str(column),
            series=series,
            non_null_count=non_null_count,
            unique_ratio=unique_ratio,
            notes=notes,
        )
        flags = _build_flags(semantic_type, unique_count=unique_count, non_null_count=non_null_count, column_name=str(column))

        if missing_ratio >= SETTINGS.profile_high_missing_ratio_threshold:
            notes.append(f"缺失率较高（{missing_ratio:.1%}）。")

        if semantic_type == "categorical" and unique_ratio >= SETTINGS.profile_high_cardinality_ratio_threshold:
            notes.append("分类列基数较高，后续建模需谨慎编码。")

        entry = {
            "column_name": str(column),
            "dtype_raw": str(series.dtype),
            "non_null_count": non_null_count,
            "missing_count": missing_count,
            "missing_ratio": missing_ratio,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "sample_values": sample_values,
            "semantic_type": semantic_type,
            "usable_for_analysis": flags["usable_for_analysis"],
            "usable_for_groupby": flags["usable_for_groupby"],
            "usable_for_ml_feature": flags["usable_for_ml_feature"],
            "usable_as_target_candidate": flags["usable_as_target_candidate"],
            "notes": notes,
        }
        columns.append(entry)
        if notes:
            warnings.append(f"{column}: {'; '.join(notes)}")

    return {
        "column_count": int(len(df.columns)),
        "row_count": total_rows,
        "columns": columns,
        "warnings": warnings[:50],
    }


def _sample_values(series: pd.Series) -> list[Any]:
    values = series.dropna()
    if values.empty:
        return []
    sample = values.head(SETTINGS.profile_sample_values_count).tolist()
    return [_to_plain_scalar(item) for item in sample]


def _infer_semantic_type(
    *,
    column_name: str,
    series: pd.Series,
    non_null_count: int,
    unique_ratio: float,
    notes: list[str],
) -> str:
    lower_name = column_name.strip().lower()
    if non_null_count == 0:
        notes.append("列全为空。")
        return "unknown"

    if pd.api.types.is_numeric_dtype(series):
        if int(series.nunique(dropna=True)) == 2:
            notes.append("二值列，可能可用作标签列。")
            return "binary_label_candidate"
        return "numeric"

    values = series.dropna().astype(str).str.strip()
    if values.empty:
        notes.append("列全为空白字符串。")
        return "unknown"

    unique_count = int(values.nunique(dropna=True))
    if unique_count == 2:
        notes.append("二值分类列，可能可用作标签列。")
        return "binary_label_candidate"

    datetime_ratio = _datetime_parse_ratio(values)
    if datetime_ratio >= SETTINGS.profile_datetime_parse_ratio_threshold:
        notes.append(f"检测到日期时间模式（可解析比例 {datetime_ratio:.1%}）。")
        return "datetime_like"

    avg_len = float(values.str.len().mean())
    if avg_len >= SETTINGS.profile_text_avg_length_threshold and unique_ratio >= SETTINGS.profile_high_cardinality_ratio_threshold:
        notes.append("文本长度较长且重复度低，疑似自由文本列。")
        return "text_like"
    if any(hint in lower_name for hint in TEXT_NAME_HINTS):
        notes.append("列名提示为文本描述字段。")
        return "text_like"

    is_identifier_name = any(hint in lower_name for hint in IDENTIFIER_NAME_HINTS)
    if is_identifier_name or (
        unique_ratio >= SETTINGS.profile_identifier_unique_ratio_threshold
        and avg_len < SETTINGS.profile_text_avg_length_threshold
    ):
        notes.append("高唯一性，疑似标识符列。")
        return "identifier_like"

    return "categorical"


def _build_flags(semantic_type: str, *, unique_count: int, non_null_count: int, column_name: str) -> dict[str, bool]:
    lower_name = column_name.strip().lower()
    target_name_hint = any(hint in lower_name for hint in TARGET_NAME_HINTS)

    usable_for_analysis = semantic_type in {"numeric", "categorical", "datetime_like", "binary_label_candidate"}
    usable_for_groupby = semantic_type in {"categorical", "datetime_like", "binary_label_candidate"}
    usable_for_ml_feature = semantic_type in {"numeric", "categorical", "datetime_like", "binary_label_candidate"}
    if semantic_type in {"identifier_like", "text_like", "unknown"}:
        usable_for_ml_feature = False

    usable_as_target_candidate = False
    if semantic_type == "binary_label_candidate":
        usable_as_target_candidate = True
    elif semantic_type == "categorical" and 2 <= unique_count <= 10:
        usable_as_target_candidate = True
    elif semantic_type == "numeric" and 2 <= unique_count <= 20 and non_null_count > 10:
        usable_as_target_candidate = True
    if target_name_hint and semantic_type not in {"identifier_like", "unknown"}:
        usable_as_target_candidate = True

    return {
        "usable_for_analysis": usable_for_analysis,
        "usable_for_groupby": usable_for_groupby,
        "usable_for_ml_feature": usable_for_ml_feature,
        "usable_as_target_candidate": usable_as_target_candidate,
    }


def _datetime_parse_ratio(values: pd.Series) -> float:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if len(values.index) == 0:
        return 0.0
    return float(parsed.notna().sum()) / float(len(values.index))


def _to_plain_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value
