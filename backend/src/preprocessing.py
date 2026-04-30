from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.settings import SETTINGS


class ModelPrepPlanError(ValueError):
    pass


@dataclass(slots=True)
class ModelInputBundle:
    x: pd.DataFrame
    y: pd.Series
    preprocessor: ColumnTransformer
    target: str
    features_used: list[str]
    excluded_columns: list[dict[str, str]]
    feature_names_after_transform: list[str]
    prep_summary: dict[str, Any]
    warnings: list[str]


def prepare_analysis_dataframe(df: pd.DataFrame, profile_artifact: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    analysis_df = df.copy(deep=True)
    steps: list[dict[str, Any]] = []
    warnings: list[str] = []
    excluded_columns: list[dict[str, str]] = []
    retained_columns: list[str] = []

    profile_columns = _profile_column_map(profile_artifact)

    all_null_columns = [col for col in analysis_df.columns if int(analysis_df[col].notna().sum()) == 0]
    if all_null_columns:
        analysis_df = analysis_df.drop(columns=all_null_columns)
        steps.append({"type": "drop_all_null_columns", "columns": [str(col) for col in all_null_columns]})
        for col in all_null_columns:
            excluded_columns.append({"column": str(col), "reason": "all_null"})

    for column in list(analysis_df.columns):
        meta = profile_columns.get(str(column), {})
        semantic_type = str(meta.get("semantic_type", "unknown"))
        series = analysis_df[column]

        if semantic_type == "numeric" and not pd.api.types.is_numeric_dtype(series):
            converted = pd.to_numeric(series, errors="coerce")
            if converted.notna().sum() > 0:
                analysis_df[column] = converted
                steps.append({"type": "coerce_numeric", "columns": [str(column)]})

        elif semantic_type == "datetime_like":
            converted = pd.to_datetime(series, errors="coerce", format="mixed")
            if converted.notna().sum() > 0:
                analysis_df[column] = converted
                steps.append({"type": "coerce_datetime", "columns": [str(column)]})
            else:
                warnings.append(f"{column} 被识别为日期列，但无法稳定转换为 datetime。")

        if semantic_type in {"identifier_like", "text_like"}:
            warnings.append(f"{column} 为 {semantic_type}，建议仅用于检索或说明，不建议直接建模。")
        retained_columns.append(str(column))

    if not steps:
        steps.append({"type": "noop", "reason": "analysis阶段未执行必要的类型转换。"})

    payload = {
        "stage": "analysis",
        "steps": steps,
        "retained_columns": retained_columns,
        "excluded_columns": excluded_columns,
        "row_count": int(len(analysis_df.index)),
        "column_count": int(len(analysis_df.columns)),
    }
    return analysis_df, payload, warnings


def plan_model_preprocessing(
    df: pd.DataFrame,
    profile_artifact: dict[str, Any],
    *,
    target: str | None = None,
    features: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    profile_columns = _profile_column_map(profile_artifact)
    warnings: list[str] = []
    excluded_columns: list[dict[str, str]] = []

    if target is not None and target not in df.columns:
        raise ModelPrepPlanError(f"目标列不存在: {target}")

    target_status = "not_provided"
    selected_target: str | None = target

    if target is None:
        candidates = [
            col
            for col, meta in profile_columns.items()
            if bool(meta.get("usable_as_target_candidate"))
        ]
        if len(candidates) == 1:
            selected_target = candidates[0]
            target_status = "inferred_single"
        elif len(candidates) > 1:
            target_status = "ambiguous"
            selected_target = None
            warnings.append(f"检测到多个目标候选列：{', '.join(candidates[:SETTINGS.model_prep_max_target_candidates])}。")
        else:
            target_status = "not_found"
            warnings.append("未检测到明确目标候选列，请显式指定 target。")
    else:
        target_meta = profile_columns.get(target, {})
        target_semantic = str(target_meta.get("semantic_type", "unknown"))
        if target_semantic in {"identifier_like", "text_like", "unknown"}:
            raise ModelPrepPlanError(f"目标列不适合建模: {target}（{target_semantic}）")
        target_status = "explicit_validated"

    requested_features: list[str]
    if features is not None:
        for feature in features:
            if feature not in df.columns:
                raise ModelPrepPlanError(f"特征列不存在: {feature}")
        requested_features = list(features)
    else:
        requested_features = [str(col) for col in df.columns]

    candidate_features: list[str] = []
    feature_types: dict[str, list[str]] = {"numeric": [], "categorical": [], "datetime_like": [], "binary_label_candidate": []}

    for feature in requested_features:
        if selected_target and feature == selected_target:
            excluded_columns.append({"column": feature, "reason": "target_column"})
            continue

        meta = profile_columns.get(feature, {})
        semantic_type = str(meta.get("semantic_type", "unknown"))
        if semantic_type in {"identifier_like", "text_like", "unknown"}:
            excluded_columns.append({"column": feature, "reason": semantic_type})
            continue

        unique_count = int(meta.get("unique_count", 0))
        non_null_count = int(meta.get("non_null_count", 0))
        unique_ratio = float(meta.get("unique_ratio", 0.0))
        if non_null_count > 0 and unique_count == non_null_count and unique_ratio >= SETTINGS.profile_identifier_unique_ratio_threshold:
            excluded_columns.append({"column": feature, "reason": "near_unique"})
            continue

        candidate_features.append(feature)
        if semantic_type in feature_types:
            feature_types[semantic_type].append(feature)
        elif semantic_type == "numeric":
            feature_types["numeric"].append(feature)
        else:
            feature_types["categorical"].append(feature)

    if not candidate_features:
        warnings.append("未找到可用特征列，请显式指定 features 并检查列语义。")

    payload = {
        "target": selected_target,
        "target_status": target_status,
        "candidate_features": candidate_features,
        "excluded_columns": excluded_columns,
        "feature_types": feature_types,
        "feature_count": int(len(candidate_features)),
        "row_count": int(len(df.index)),
    }
    return payload, warnings


def prepare_model_inputs(
    df: pd.DataFrame,
    profile_artifact: dict[str, Any],
    model_prep_plan: dict[str, Any],
    *,
    target: str | None = None,
    features: list[str] | None = None,
) -> ModelInputBundle:
    warnings: list[str] = []
    profile_columns = _profile_column_map(profile_artifact)
    excluded_columns: list[dict[str, str]] = []
    plan_excluded = model_prep_plan.get("excluded_columns", [])
    if isinstance(plan_excluded, list):
        for item in plan_excluded:
            if isinstance(item, dict) and "column" in item and "reason" in item:
                excluded_columns.append({"column": str(item["column"]), "reason": str(item["reason"])})

    selected_target = target or model_prep_plan.get("target")
    if not isinstance(selected_target, str) or selected_target not in df.columns:
        raise ModelPrepPlanError("未找到可用目标列，请先在 model_prep_plan 中明确 target。")

    plan_features = model_prep_plan.get("candidate_features")
    if features is not None:
        for feature in features:
            if feature not in df.columns:
                raise ModelPrepPlanError(f"特征列不存在: {feature}")
        candidate_features = list(features)
    elif isinstance(plan_features, list):
        candidate_features = [str(item) for item in plan_features if isinstance(item, str)]
    else:
        candidate_features = []

    if not candidate_features:
        raise ModelPrepPlanError("没有可用于建模的特征列。")

    numeric_features: list[str] = []
    categorical_features: list[str] = []
    usable_features: list[str] = []

    for feature in candidate_features:
        if feature == selected_target:
            excluded_columns.append({"column": feature, "reason": "target_column"})
            continue
        meta = profile_columns.get(feature, {})
        semantic_type = str(meta.get("semantic_type", "unknown"))
        if semantic_type in {"identifier_like", "text_like", "unknown"}:
            excluded_columns.append({"column": feature, "reason": semantic_type})
            continue
        if semantic_type == "datetime_like":
            excluded_columns.append({"column": feature, "reason": "datetime_like"})
            warnings.append(f"{feature} 为 datetime_like，当前 baseline ML 阶段默认排除。")
            continue

        usable_features.append(feature)
        if semantic_type == "numeric":
            numeric_features.append(feature)
        else:
            categorical_features.append(feature)

    if len(usable_features) > SETTINGS.ml_max_feature_count:
        warnings.append(f"特征数超过上限 {SETTINGS.ml_max_feature_count}，仅保留前 {SETTINGS.ml_max_feature_count} 个。")
        usable_features = usable_features[: SETTINGS.ml_max_feature_count]
        numeric_features = [item for item in numeric_features if item in usable_features]
        categorical_features = [item for item in categorical_features if item in usable_features]

    if not usable_features:
        raise ModelPrepPlanError("过滤后没有可用特征列，无法训练 baseline 模型。")

    data = df.loc[:, usable_features + [selected_target]].copy(deep=True)
    valid_mask = data[selected_target].notna()
    dropped_target_na = int((~valid_mask).sum())
    if dropped_target_na > 0:
        warnings.append(f"目标列存在缺失，已丢弃 {dropped_target_na} 行。")
    data = data.loc[valid_mask]
    if data.empty:
        raise ModelPrepPlanError("目标列有效样本为空，无法训练模型。")

    if len(data.index) > SETTINGS.ml_max_training_rows:
        warnings.append(f"样本数超过上限 {SETTINGS.ml_max_training_rows}，已截断为前 {SETTINGS.ml_max_training_rows} 行。")
        data = data.head(SETTINGS.ml_max_training_rows).copy(deep=True)

    x = data.loc[:, usable_features].copy(deep=True)
    y = data[selected_target].copy(deep=True)

    for feature in numeric_features:
        x[feature] = pd.to_numeric(x[feature], errors="coerce")
    for feature in categorical_features:
        # Fill nulls first; avoid converting real missing values into literal "nan".
        x[feature] = x[feature].fillna("Unknown").astype(str)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=SETTINGS.ml_max_ohe_categories,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))
    if not transformers:
        raise ModelPrepPlanError("当前数据不包含可用于 baseline ML 的数值/分类特征。")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    preprocessor.fit(x)
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [str(col) for col in usable_features]
        warnings.append("未能稳定提取转换后特征名，将使用原始列名近似表示。")

    prep_summary = {
        "target": selected_target,
        "feature_count_before_filter": int(len(candidate_features)),
        "feature_count_after_filter": int(len(usable_features)),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "rows_used": int(len(x.index)),
        "dropped_target_null_rows": dropped_target_na,
        "imputation": {"numeric": "median", "categorical": "Unknown"},
        "encoding": {"categorical": "one_hot_ignore_unknown"},
    }
    return ModelInputBundle(
        x=x,
        y=y,
        preprocessor=preprocessor,
        target=selected_target,
        features_used=usable_features,
        excluded_columns=excluded_columns,
        feature_names_after_transform=feature_names,
        prep_summary=prep_summary,
        warnings=warnings,
    )


def infer_positive_label(series: pd.Series, *, explicit_label: Any | None = None) -> dict[str, Any]:
    if explicit_label is not None:
        return {"positive_label": explicit_label, "source": "explicit", "warning": None}

    non_null = series.dropna()
    if non_null.empty:
        return {"positive_label": None, "source": "ambiguous", "warning": "目标列为空，无法推断正类标签。"}

    if pd.api.types.is_bool_dtype(non_null):
        return {"positive_label": True, "source": "boolean_default", "warning": None}

    if pd.api.types.is_numeric_dtype(non_null):
        numeric_values = pd.to_numeric(non_null, errors="coerce").dropna()
        unique_numeric = sorted(set(numeric_values.tolist()))
        if unique_numeric == [0, 1] or unique_numeric == [0.0, 1.0]:
            return {"positive_label": 1, "source": "binary_numeric_default", "warning": None}
        if len(unique_numeric) == 2:
            return {
                "positive_label": max(unique_numeric),
                "source": "binary_numeric_default",
                "warning": f"目标列为二值数值 {unique_numeric}，默认取较大值为正类。",
            }
        return {"positive_label": None, "source": "ambiguous", "warning": "目标列不是稳定二值标签，无法自动推断正类。"}

    normalized = non_null.astype(str).str.strip()
    unique_text = [item for item in sorted(normalized.unique().tolist()) if item != ""]
    if len(unique_text) != 2:
        return {"positive_label": None, "source": "ambiguous", "warning": f"目标标签共有 {len(unique_text)} 类，当前仅支持二分类。"}

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

    return {"positive_label": None, "source": "ambiguous", "warning": "未能可靠推断正类，请显式提供 positive_label。"}


def _profile_column_map(profile_artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    columns = profile_artifact.get("columns", [])
    if not isinstance(columns, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in columns:
        if isinstance(item, dict):
            name = item.get("column_name")
            if isinstance(name, str):
                result[name] = item
    return result
