from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

INTENT_PLANNER_MODEL: Any | None = None
DEFAULT_INTENT_PLANNER_MODEL = os.getenv(
    "INTENT_PLANNER_MODEL",
    os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
)


class IntentInterpretationPayload(BaseModel):
    intent_type: Literal["analysis", "ml", "chart", "mixed", "followup"]
    is_dataset_overview: bool = False
    is_follow_up: bool = False
    requires_ml: bool = False
    requires_chart: bool = False
    requires_python_analysis: bool = False
    confidence: Literal["low", "medium", "high"] = "medium"
    conflict_flags: list[str] = Field(default_factory=list)
    route_source: Literal["llm_primary", "llm_with_guardrail", "heuristic_fallback"] = "llm_primary"
    deliverables: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    suggested_plan: list[str] = Field(default_factory=list)


class CoarseIntentPayload(BaseModel):
    primary_intent: Literal["dataset_overview", "followup", "analysis", "chart", "ml", "mixed"]
    confidence: Literal["low", "medium", "high"] = "medium"
    reasoning_summary: str = ""


COARSE_INTENT_SYSTEM_PROMPT = (
    "你是 Data Agent 的粗粒度路由解释器。"
    "你的任务不是执行分析，而是先把用户请求归类到一个主路由。"
    "只输出 JSON，不要输出 markdown、解释文本或多余前后缀。"
    "必须输出这些字段：primary_intent, confidence, reasoning_summary。"
    "primary_intent 只能是 dataset_overview, followup, analysis, chart, ml, mixed 之一。"
    "confidence 只能是 low, medium, high 之一。"
    "判定规则："
    "1) 如果用户主要是在要求讲解、介绍、概览当前数据集，而不是立即执行深入分析，优先判定为 dataset_overview。"
    "2) 如果用户明显在引用上一个结果、上一个模型、刚才的图或之前的结论，优先判定为 followup。"
    "3) 只有当用户明确要求训练、建模、预测、模型评估、特征重要性，且目标是从训练好的模型中得到结果时，才判定为 ml。"
    "4) 只是在分析、比较、分组、探索、描述、看关系或驱动因素时，优先判定为 analysis 或 mixed，不要直接判定成 ml。"
    "5) 如果同时需要分析和建模，判定为 mixed。"
    "6) 如果核心目标是画图、可视化、图表，可判定为 chart；若同时需要明显分析步骤，也可判定 mixed。"
)

DETAIL_INTENT_SYSTEM_PROMPT = (
    "你是 Data Agent 的细粒度路由解释器。"
    "你的任务不是执行分析，而是根据给定主路由补全结构化执行意图。"
    "只输出 JSON，不要输出 markdown、解释文本或多余前后缀。"
    "必须输出这些字段："
    "intent_type, is_dataset_overview, is_follow_up, requires_ml, requires_chart, requires_python_analysis, "
    "deliverables, reasoning_summary, suggested_plan。"
    "intent_type 只能是 analysis, ml, chart, mixed, followup 之一。"
    "判定规则："
    "1) dataset_overview 场景下，is_dataset_overview 设为 true，intent_type 通常设为 followup。"
    "2) 只有当用户明确要求训练、建模、预测、模型评估、特征重要性时，才把 requires_ml 设为 true。"
    "3) 只是在分析、比较、分组、探索、描述、看关系或驱动因素时，优先判定为 analysis 或 mixed，不要直接判定成 ml。"
    "4) 如果同时需要分析和建模，判定为 mixed，并给出先 analysis 后 ml 的 suggested_plan。"
    "5) 如果用户明显在引用上一个结果、上一个模型、刚才的图或之前的结论，is_follow_up 设为 true，通常 intent_type 判定为 followup。"
    "6) 如果用户要求画图、可视化、图表，requires_chart 设为 true。"
    "deliverables 只能使用这些值或其子集：summary, metrics, feature_importance, chart, table, prediction, explanation。"
)


def _build_model() -> Any | None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return ChatDeepSeek(
        model=DEFAULT_INTENT_PLANNER_MODEL,
        temperature=0,
        api_key=api_key,
        api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
    )


def get_intent_planner_model() -> Any | None:
    global INTENT_PLANNER_MODEL
    if INTENT_PLANNER_MODEL is not None:
        return INTENT_PLANNER_MODEL
    INTENT_PLANNER_MODEL = _build_model()
    return INTENT_PLANNER_MODEL


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
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _normalize_deliverables(values: list[str] | None) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        token = value.strip().lower().replace(" ", "_").strip(".,;:，。；：")
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _normalize_plan(values: list[str] | None) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        raw_items = re.split(r"[\r\n]+|(?<=\.)\s+(?=\d+[\.\)])|(?<=\))\s+", values)
        values = [item.strip(" -•\t") for item in raw_items if item.strip(" -•\t")]
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        token = re.sub(r"^\d+[\.\)]\s*", "", value.strip())
        token = token.strip(" -•\t")
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _normalize_conflict_flags(values: list[str] | None) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        token = value.strip().lower().replace(" ", "_")
        token = token.strip(" -•\t.,;:，。；：")
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _coerce_payload(payload: dict[str, Any]) -> IntentInterpretationPayload | None:
    try:
        return IntentInterpretationPayload.model_validate(
            {
                "intent_type": payload.get("intent_type"),
                "is_dataset_overview": payload.get("is_dataset_overview", False),
                "is_follow_up": payload.get("is_follow_up", False),
                "requires_ml": payload.get("requires_ml", False),
                "requires_chart": payload.get("requires_chart", False),
                "requires_python_analysis": payload.get("requires_python_analysis", False),
                "confidence": payload.get("confidence", "medium"),
                "conflict_flags": _normalize_conflict_flags(payload.get("conflict_flags")),
                "route_source": payload.get("route_source", "llm_primary"),
                "deliverables": _normalize_deliverables(payload.get("deliverables")),
                "reasoning_summary": str(payload.get("reasoning_summary", "")).strip(),
                "suggested_plan": _normalize_plan(payload.get("suggested_plan")),
            }
        )
    except ValidationError as exc:
        logger.debug("intent planner payload validation failed: %s", exc)
        return None


def _coerce_coarse_payload(payload: dict[str, Any]) -> CoarseIntentPayload | None:
    try:
        return CoarseIntentPayload.model_validate(
            {
                "primary_intent": payload.get("primary_intent"),
                "confidence": payload.get("confidence", "medium"),
                "reasoning_summary": str(payload.get("reasoning_summary", "")).strip(),
            }
        )
    except ValidationError as exc:
        logger.debug("coarse intent planner payload validation failed: %s", exc)
        return None


def _build_messages(
    message: str,
    *,
    dataset_columns: list[str],
    prior_analysis_active: bool,
    system_prompt: str,
    extra_payload: dict[str, Any] | None = None,
) -> list[Any]:
    prompt_payload = {
        "message": message,
        "dataset_columns": dataset_columns,
        "prior_analysis_active": prior_analysis_active,
    }
    if extra_payload:
        prompt_payload.update(extra_payload)
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(prompt_payload, ensure_ascii=False)),
    ]


def _invoke_model_json(
    model: Any,
    *,
    message: str,
    dataset_columns: list[str],
    prior_analysis_active: bool,
    system_prompt: str,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        response = model.invoke(
            _build_messages(
                message,
                dataset_columns=dataset_columns,
                prior_analysis_active=prior_analysis_active,
                system_prompt=system_prompt,
                extra_payload=extra_payload,
            )
        )
    except Exception as exc:  # pragma: no cover - network/provider errors are fallback paths
        logger.debug("intent planner model invocation failed: %s", exc)
        return None

    text = _extract_text(response)
    if not text.strip():
        return None

    try:
        payload = json.loads(_extract_json_candidate(text))
    except Exception as exc:
        logger.debug("intent planner JSON parsing failed: %s", exc)
        return None
    return payload if isinstance(payload, dict) else None


def plan_intent_with_llm(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
) -> IntentInterpretationPayload | None:
    model = get_intent_planner_model()
    if model is None:
        return None
    resolved_dataset_columns = dataset_columns or []

    coarse_raw = _invoke_model_json(
        model,
        message=message,
        dataset_columns=resolved_dataset_columns,
        prior_analysis_active=prior_analysis_active,
        system_prompt=COARSE_INTENT_SYSTEM_PROMPT,
    )
    if coarse_raw is None:
        return None
    coarse_payload = _coerce_coarse_payload(coarse_raw)
    if coarse_payload is None:
        return None

    detail_raw = _invoke_model_json(
        model,
        message=message,
        dataset_columns=resolved_dataset_columns,
        prior_analysis_active=prior_analysis_active,
        system_prompt=DETAIL_INTENT_SYSTEM_PROMPT,
        extra_payload={
            "coarse_intent": coarse_payload.primary_intent,
            "coarse_confidence": coarse_payload.confidence,
            "coarse_reasoning_summary": coarse_payload.reasoning_summary,
        },
    )
    if detail_raw is None:
        return None

    detail_raw.setdefault("confidence", coarse_payload.confidence)
    detail_raw.setdefault("conflict_flags", [])
    detail_raw.setdefault("route_source", "llm_primary")
    if coarse_payload.primary_intent == "dataset_overview":
        detail_raw["is_dataset_overview"] = True
        detail_raw.setdefault("intent_type", "followup")
    elif coarse_payload.primary_intent == "followup":
        detail_raw["is_follow_up"] = True
        detail_raw.setdefault("intent_type", "followup")
    elif coarse_payload.primary_intent in {"analysis", "chart", "ml", "mixed"}:
        detail_raw.setdefault("intent_type", coarse_payload.primary_intent)
    if not detail_raw.get("reasoning_summary") and coarse_payload.reasoning_summary:
        detail_raw["reasoning_summary"] = coarse_payload.reasoning_summary

    return _coerce_payload(detail_raw)
