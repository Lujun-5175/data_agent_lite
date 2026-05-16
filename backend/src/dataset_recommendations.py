from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_MAX_PROMPTS = 3


def generate_recommended_prompts(
    *,
    dataset_summary: dict[str, Any],
    schema_profile: dict[str, Any],
) -> list[str]:
    prompts = _generate_prompts_with_model(
        dataset_summary=dataset_summary,
        schema_profile=schema_profile,
    )
    if prompts:
        return prompts
    return build_fallback_recommended_prompts(
        dataset_summary=dataset_summary,
        schema_profile=schema_profile,
    )


def _generate_prompts_with_model(
    *,
    dataset_summary: dict[str, Any],
    schema_profile: dict[str, Any],
) -> list[str]:
    from src.intent_planner import get_intent_planner_model

    model = get_intent_planner_model()
    if model is None:
        return []

    payload = {
        "dataset_summary": {
            "filename": dataset_summary.get("filename"),
            "analysis_basis": dataset_summary.get("analysis_basis"),
            "row_count": dataset_summary.get("row_count"),
            "column_count": dataset_summary.get("column_count"),
            "numeric_columns": dataset_summary.get("numeric_columns", []),
            "categorical_columns": dataset_summary.get("categorical_columns", []),
            "warnings": dataset_summary.get("warnings", []),
        },
        "schema_profile": {
            "columns": schema_profile.get("columns", []),
            "warnings": schema_profile.get("warnings", []),
        },
    }
    messages = [
        SystemMessage(
            content=(
                "你是 Data Agent 的推荐问题生成器。"
                "请基于给定数据集摘要和 schema profile，生成 3 个简洁、可执行、彼此不同的问题。"
                "问题必须面向当前数据集，优先覆盖趋势、分组比较、关系/检验/建模中的高价值入口。"
                "若数据不适合建模，请不要强行给建模问题。"
                "只输出 JSON，格式为 {\"recommended_prompts\": [\"...\", \"...\", \"...\"]}。"
            )
        ),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    try:
        response = model.invoke(messages)
    except Exception as exc:
        logger.debug("recommended prompt model invocation failed: %s", exc)
        return []
    text = _extract_text(response)
    if not text.strip():
        return []
    try:
        parsed = json.loads(_extract_json_candidate(text))
    except Exception as exc:
        logger.debug("recommended prompt JSON parsing failed: %s", exc)
        return []
    if not isinstance(parsed, dict):
        return []
    return _normalize_prompt_list(parsed.get("recommended_prompts"))


def build_fallback_recommended_prompts(
    *,
    dataset_summary: dict[str, Any],
    schema_profile: dict[str, Any],
) -> list[str]:
    numeric_columns = [str(item) for item in dataset_summary.get("numeric_columns", []) if isinstance(item, str)]
    categorical_columns = [str(item) for item in dataset_summary.get("categorical_columns", []) if isinstance(item, str)]
    schema_columns = schema_profile.get("columns", [])

    datetime_columns = [
        str(item.get("column_name"))
        for item in schema_columns
        if isinstance(item, dict) and item.get("semantic_type") == "datetime_like" and item.get("column_name")
    ]
    target_candidates = [
        str(item.get("column_name"))
        for item in schema_columns
        if isinstance(item, dict) and item.get("usable_as_target_candidate") and item.get("column_name")
    ]

    prompts: list[str] = []
    if datetime_columns and numeric_columns:
        prompts.append(f"请按 {datetime_columns[0]} 聚合，分析 {numeric_columns[0]} 的趋势，并画一张折线图。")
    if categorical_columns and numeric_columns:
        prompts.append(f"按 {categorical_columns[0]} 分组比较 {numeric_columns[0]} 的差异，并给出结论。")
    if len(numeric_columns) >= 2:
        prompts.append(f"{numeric_columns[0]} 和 {numeric_columns[1]} 的相关性是多少？")

    for target_column in target_candidates[:1]:
        if target_column in numeric_columns:
            feature_candidates = [column for column in numeric_columns if column != target_column]
            if feature_candidates:
                prompts.append(
                    f"用 {', '.join(feature_candidates[:2])} 预测 {target_column}，跑一个线性回归并汇报指标。"
                )
                break
        else:
            feature_candidates = [column for column in [*numeric_columns, *categorical_columns] if column != target_column]
            if feature_candidates:
                prompts.append(
                    f"用 {', '.join(feature_candidates[:3])} 预测 {target_column}，尝试一个 baseline 分类模型并汇报指标。"
                )
                break

    prompts.extend(
        [
            "请先做一份描述性统计，并指出最值得关注的字段。",
            "按一个关键分类字段分组，比较主要指标差异。",
            "哪些字段最适合继续做相关性、检验或建模分析？",
        ]
    )
    return _normalize_prompt_list(prompts)


def _extract_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            nested = getattr(item, "content", None)
            if isinstance(nested, str):
                parts.append(nested)
        return "".join(parts)
    return str(content)


def _extract_json_candidate(text: str) -> str:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _normalize_prompt_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = " ".join(item.strip().split())
        if len(text) < 6 or text in normalized:
            continue
        normalized.append(text)
        if len(normalized) >= _MAX_PROMPTS:
            break
    return normalized
