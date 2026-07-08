from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from src.settings import SETTINGS

IDENTIFIER_NAME_HINTS = ("id", "uuid", "key", "code", "number", "编号", "账号")
TEXT_NAME_HINTS = ("note", "comment", "remark", "desc", "text", "说明", "备注")
TARGET_NAME_HINTS = ("target", "label", "churn", "outcome", "是否")
_PATTERN_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PATTERN_CODE_RE = re.compile(r"^[A-Za-z]*[-_]?\d+[A-Za-z0-9_-]*$")
_PATTERN_DATE_TOKEN_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")


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
        semantic_evidence: list[str] = []
        value_shape_summary = _collect_value_shape_summary(series)
        cardinality_metrics = _build_cardinality_metrics(series)
        pattern_hints = _collect_pattern_hints(series)

        semantic_type = _infer_semantic_type(
            column_name=str(column),
            series=series,
            non_null_count=non_null_count,
            unique_ratio=unique_ratio,
            notes=notes,
            semantic_evidence=semantic_evidence,
            value_shape_summary=value_shape_summary,
            cardinality_metrics=cardinality_metrics,
            pattern_hints=pattern_hints,
        )
        flags = _build_flags(semantic_type, unique_count=unique_count, non_null_count=non_null_count, column_name=str(column))

        if missing_ratio >= SETTINGS.profile_high_missing_ratio_threshold:
            notes.append(f"High missing rate ({missing_ratio:.1%}). ")
            semantic_evidence.append("high_missing_ratio")

        if semantic_type == "categorical" and unique_ratio >= SETTINGS.profile_high_cardinality_ratio_threshold:
            notes.append("Categorical column has high cardinality. Use caution with encoding for modeling.")
            semantic_evidence.append("high_cardinality_categorical")

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
            "semantic_evidence": semantic_evidence,
            "value_shape_summary": value_shape_summary,
            "cardinality_metrics": cardinality_metrics,
            "pattern_hints": pattern_hints,
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
    semantic_evidence: list[str],
    value_shape_summary: dict[str, float],
    cardinality_metrics: dict[str, float | int | None],
    pattern_hints: list[str],
) -> str:
    lower_name = column_name.strip().lower()
    if non_null_count == 0:
        notes.append("Column is entirely empty。")
        semantic_evidence.append("all_null")
        return "unknown"

    if pd.api.types.is_numeric_dtype(series):
        if int(series.nunique(dropna=True)) == 2:
            notes.append("二值列，可能可用作标签列。")
            semantic_evidence.extend(["numeric_dtype", "binary_unique_count"])
            return "binary_label_candidate"
        semantic_evidence.append("numeric_dtype")
        return "numeric"

    values = series.dropna().astype(str).str.strip()
    if values.empty:
        notes.append("Column is entirely empty白字符串。")
        semantic_evidence.append("blank_string_only")
        return "unknown"

    unique_count = int(values.nunique(dropna=True))
    if unique_count == 2:
        notes.append("二值分类列，可能可用作标签列。")
        semantic_evidence.append("binary_unique_count")
        return "binary_label_candidate"

    datetime_ratio = _datetime_parse_ratio(values)
    if datetime_ratio >= SETTINGS.profile_datetime_parse_ratio_threshold:
        notes.append(f"Date/time pattern detected（可解析比例 {datetime_ratio:.1%}). ")
        semantic_evidence.extend(["datetime_parse_ratio_high", f"datetime_ratio={datetime_ratio:.2f}"])
        return "datetime_like"

    avg_len = float(values.str.len().mean())
    entropy = float(cardinality_metrics.get("normalized_entropy") or 0.0)
    contains_space_ratio = float(value_shape_summary.get("contains_space", 0.0))
    alpha_ratio = float(value_shape_summary.get("alpha_only", 0.0))
    code_like_ratio = float(value_shape_summary.get("code_like", 0.0))
    is_identifier_name = any(hint in lower_name for hint in IDENTIFIER_NAME_HINTS)

    if any(hint in lower_name for hint in TEXT_NAME_HINTS):
        notes.append("Column name suggests text/description field。")
        semantic_evidence.append("text_name_hint")
        return "text_like"

    if avg_len >= SETTINGS.profile_text_avg_length_threshold and unique_ratio >= SETTINGS.profile_high_cardinality_ratio_threshold:
        notes.append("Long text length with low repetition，Likely free-text column。")
        semantic_evidence.extend(["long_average_text", "high_unique_ratio"])
        return "text_like"

    if contains_space_ratio >= 0.45 and unique_ratio >= 0.35:
        notes.append("Values contain many spaces with high dispersion，Likely free-text column。")
        semantic_evidence.extend(["contains_space_ratio_high", "moderate_to_high_unique_ratio"])
        return "text_like"

    if is_identifier_name or code_like_ratio >= 0.6 or (
        unique_ratio >= SETTINGS.profile_identifier_unique_ratio_threshold
        and avg_len < SETTINGS.profile_text_avg_length_threshold
    ):
        notes.append("High uniqueness or clear encoding pattern，Likely identifier column。")
        semantic_evidence.extend(["identifier_like_signal", f"code_like_ratio={code_like_ratio:.2f}"])
        return "identifier_like"

    if alpha_ratio >= 0.8 and entropy < 0.65:
        semantic_evidence.append("low_entropy_alpha_tokens")
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


def _collect_value_shape_summary(series: pd.Series) -> dict[str, float]:
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return {}

    total = float(len(values.index))
    counts = {
        "digit_only": 0,
        "alpha_only": 0,
        "alnum": 0,
        "contains_hyphen": 0,
        "contains_space": 0,
        "code_like": 0,
    }
    for value in values:
        if value.isdigit():
            counts["digit_only"] += 1
        if value.isalpha():
            counts["alpha_only"] += 1
        if value.isalnum():
            counts["alnum"] += 1
        if "-" in value or "_" in value:
            counts["contains_hyphen"] += 1
        if " " in value:
            counts["contains_space"] += 1
        if _PATTERN_CODE_RE.match(value):
            counts["code_like"] += 1
    return {key: round(value / total, 4) for key, value in counts.items()}


def _build_cardinality_metrics(series: pd.Series) -> dict[str, float | int | None]:
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return {"top_value_ratio": None, "normalized_entropy": None}

    counts = values.value_counts(dropna=True)
    probabilities = (counts / counts.sum()).tolist()
    entropy = 0.0
    for probability in probabilities:
        if probability > 0:
            entropy -= probability * math.log(probability, 2)
    max_entropy = math.log(len(counts.index), 2) if len(counts.index) > 1 else 0.0
    normalized_entropy = (entropy / max_entropy) if max_entropy > 0 else 0.0
    top_value_ratio = float(counts.iloc[0]) / float(counts.sum())
    return {
        "distinct_count": int(len(counts.index)),
        "top_value_ratio": round(top_value_ratio, 4),
        "normalized_entropy": round(normalized_entropy, 4),
    }


def _collect_pattern_hints(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return []

    hints: list[str] = []
    sample_values = values.head(SETTINGS.profile_sample_values_count).tolist()
    if any(_PATTERN_EMAIL_RE.match(value) for value in sample_values):
        hints.append("email_like")
    if any(_PATTERN_DATE_TOKEN_RE.match(value) for value in sample_values):
        hints.append("date_token_like")
    if any(_PATTERN_CODE_RE.match(value) for value in sample_values):
        hints.append("id_like_code")
    if any(" " in value and len(value) > 12 for value in sample_values):
        hints.append("free_text_like")
    return hints


def _to_plain_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value
