from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import ValidationError

from src.routing_models import ConfidenceBand, RoutingCapability, RoutingDecision, RoutingPrimaryMode

logger = logging.getLogger(__name__)

INTENT_PLANNER_MODEL: Any | None = None
DEFAULT_INTENT_PLANNER_MODEL = os.getenv(
    "INTENT_PLANNER_MODEL",
    os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
)
IntentInterpretationPayload = RoutingDecision

ROUTING_SYSTEM_PROMPT = (
    "你是 Data Agent 的结构化路由器。"
    "你的任务不是执行分析，而是把用户请求路由成一个稳定的结构化决策。"
    "只输出 JSON，不要输出 markdown、解释文本或多余前后缀。"
    "必须输出这些字段："
    "primary_mode, confidence_score, confidence_band, needs_dataset, needs_tool_execution, "
    "needs_artifact_context, requested_capabilities, ambiguity_flags, reasoning_summary, execution_plan。"
    "primary_mode 只能是 direct_answer, dataset_overview, analysis, visualization, modeling, artifact_followup, mixed, clarification。"
    "requested_capabilities 只能使用这些值或其子集："
    "summarize_dataset, inspect_schema, reuse_prior_artifact, group_analysis, stat_test, python_analysis, "
    "chart_generation, train_model, evaluate_model, feature_importance, direct_answer。"
    "confidence_score 是 0 到 1 之间的小数。"
    "confidence_band 只能是 low, medium, high。"
    "判定原则："
    "1) 普通问答、概念解释、无需数据与工具时，优先 direct_answer。"
    "2) 用户主要要求讲解当前数据集、字段、预处理与推荐方向时，优先 dataset_overview。"
    "3) 用户明显引用上一个结果、模型、图表或 artifact 时，优先 artifact_followup，并设置 needs_artifact_context=true。"
    "4) 用户要求统计分析、分组比较、探索关系、过滤聚合时，优先 analysis。"
    "5) 用户要求画图或可视化时，优先 visualization；若同时包含明显分析步骤，可判 mixed。"
    "6) 只有当用户明确要求训练模型、预测、模型评估、特征重要性时，才优先 modeling。"
    "7) 若信息明显不足以安全路由，使用 clarification。"
    "8) execution_plan 用简短步骤描述后续应做什么。"
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


def _normalize_string_list(values: Any) -> list[str]:
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


def _normalize_primary_mode(value: Any) -> RoutingPrimaryMode:
    normalized = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    mapping: dict[str, RoutingPrimaryMode] = {
        "direct_answer": "direct_answer",
        "dataset_overview": "dataset_overview",
        "analysis": "analysis",
        "visualization": "visualization",
        "chart": "visualization",
        "modeling": "modeling",
        "ml": "modeling",
        "artifact_followup": "artifact_followup",
        "followup": "artifact_followup",
        "follow_up": "artifact_followup",
        "mixed": "mixed",
        "clarification": "clarification",
    }
    return mapping.get(normalized, "analysis")


def _normalize_capabilities(values: Any) -> list[RoutingCapability]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    normalized: list[RoutingCapability] = []
    mapping: dict[str, RoutingCapability] = {
        "summarize_dataset": "summarize_dataset",
        "inspect_schema": "inspect_schema",
        "reuse_prior_artifact": "reuse_prior_artifact",
        "group_analysis": "group_analysis",
        "stat_test": "stat_test",
        "python_analysis": "python_analysis",
        "chart_generation": "chart_generation",
        "train_model": "train_model",
        "evaluate_model": "evaluate_model",
        "feature_importance": "feature_importance",
        "direct_answer": "direct_answer",
    }
    for value in values:
        token = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        capability = mapping.get(token)
        if capability and capability not in normalized:
            normalized.append(capability)
    return normalized


def _confidence_band_from_score(score: float) -> ConfidenceBand:
    if score < 0.45:
        return "low"
    if score < 0.75:
        return "medium"
    return "high"


def _confidence_score_from_band(band: str | None) -> float:
    if band == "low":
        return 0.3
    if band == "high":
        return 0.85
    return 0.6


def _normalize_confidence_score(value: Any, *, fallback_band: str | None = None) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = _confidence_score_from_band(fallback_band)
    return max(0.0, min(1.0, score))


def _normalize_confidence_band(value: Any, *, fallback_score: float | None = None) -> ConfidenceBand:
    normalized = str(value or "").strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized  # type: ignore[return-value]
    score = fallback_score if isinstance(fallback_score, float) else 0.6
    return _confidence_band_from_score(score)


def _legacy_intent_to_primary_mode(intent_type: Any) -> RoutingPrimaryMode:
    normalized = str(intent_type or "").strip().lower()
    mapping: dict[str, RoutingPrimaryMode] = {
        "dataset_overview": "dataset_overview",
        "followup": "artifact_followup",
        "analysis": "analysis",
        "chart": "visualization",
        "ml": "modeling",
        "mixed": "mixed",
    }
    return mapping.get(normalized, "analysis")


def _derive_requested_capabilities(payload: dict[str, Any], primary_mode: RoutingPrimaryMode) -> list[RoutingCapability]:
    raw_capabilities = _normalize_capabilities(payload.get("requested_capabilities"))
    if raw_capabilities:
        return raw_capabilities

    capabilities: list[RoutingCapability] = []
    if primary_mode == "direct_answer":
        capabilities.append("direct_answer")
    if primary_mode == "dataset_overview":
        capabilities.extend(["summarize_dataset", "inspect_schema"])
    if primary_mode == "artifact_followup":
        capabilities.append("reuse_prior_artifact")
    if bool(payload.get("requires_python_analysis")) or primary_mode in {"analysis", "mixed"}:
        capabilities.append("python_analysis")
    if primary_mode in {"analysis", "mixed"}:
        capabilities.append("group_analysis")
    if bool(payload.get("requires_chart")) or primary_mode == "visualization":
        capabilities.append("chart_generation")
    if bool(payload.get("requires_ml")) or primary_mode == "modeling":
        capabilities.append("train_model")
    deliverables = _normalize_deliverables(payload.get("deliverables"))
    if "metrics" in deliverables:
        capabilities.append("evaluate_model")
    if "feature_importance" in deliverables:
        capabilities.append("feature_importance")
    return list(dict.fromkeys(capabilities))


def _derive_compat_fields(
    *,
    primary_mode: RoutingPrimaryMode,
    capabilities: list[RoutingCapability],
    deliverables: list[str],
    needs_artifact_context: bool,
) -> dict[str, Any]:
    requires_ml = any(capability in {"train_model", "evaluate_model", "feature_importance"} for capability in capabilities)
    requires_chart = "chart_generation" in capabilities
    requires_python_analysis = any(
        capability in {"group_analysis", "stat_test", "python_analysis"} for capability in capabilities
    )
    is_dataset_overview = primary_mode == "dataset_overview"
    is_follow_up = primary_mode == "artifact_followup" or needs_artifact_context

    if primary_mode == "dataset_overview":
        intent_type = "dataset_overview"
    elif primary_mode == "artifact_followup":
        intent_type = "followup"
    elif primary_mode == "visualization":
        intent_type = "chart"
    elif primary_mode == "modeling":
        intent_type = "ml"
    elif primary_mode == "mixed":
        intent_type = "mixed"
    else:
        intent_type = "analysis"

    resolved_deliverables = list(deliverables)
    if "evaluate_model" in capabilities and "metrics" not in resolved_deliverables:
        resolved_deliverables.append("metrics")
    if "feature_importance" in capabilities and "feature_importance" not in resolved_deliverables:
        resolved_deliverables.append("feature_importance")
    if "chart_generation" in capabilities and "chart" not in resolved_deliverables:
        resolved_deliverables.append("chart")
    if any(capability in {"summarize_dataset", "group_analysis", "python_analysis", "direct_answer"} for capability in capabilities):
        if "summary" not in resolved_deliverables:
            resolved_deliverables.append("summary")

    return {
        "intent_type": intent_type,
        "is_dataset_overview": is_dataset_overview,
        "is_follow_up": is_follow_up,
        "requires_ml": requires_ml,
        "requires_chart": requires_chart,
        "requires_python_analysis": requires_python_analysis,
        "deliverables": resolved_deliverables,
    }


def _derive_needs_dataset(primary_mode: RoutingPrimaryMode, capabilities: list[RoutingCapability]) -> bool:
    if primary_mode in {"dataset_overview", "analysis", "visualization", "modeling", "artifact_followup", "mixed"}:
        return True
    return any(
        capability in {
            "summarize_dataset",
            "inspect_schema",
            "reuse_prior_artifact",
            "group_analysis",
            "stat_test",
            "python_analysis",
            "chart_generation",
            "train_model",
            "evaluate_model",
            "feature_importance",
        }
        for capability in capabilities
    )


def _derive_needs_tool_execution(primary_mode: RoutingPrimaryMode, capabilities: list[RoutingCapability]) -> bool:
    if primary_mode in {"analysis", "visualization", "modeling", "mixed"}:
        return True
    return any(
        capability in {
            "group_analysis",
            "stat_test",
            "python_analysis",
            "chart_generation",
            "train_model",
            "evaluate_model",
            "feature_importance",
        }
        for capability in capabilities
    )


def _coerce_payload(payload: dict[str, Any]) -> IntentInterpretationPayload | None:
    primary_mode = _normalize_primary_mode(payload.get("primary_mode") or _legacy_intent_to_primary_mode(payload.get("intent_type")))
    capabilities = _derive_requested_capabilities(payload, primary_mode)
    deliverables = _normalize_deliverables(payload.get("deliverables"))
    needs_artifact_context = bool(payload.get("needs_artifact_context") or payload.get("is_follow_up") or primary_mode == "artifact_followup")
    compatibility = _derive_compat_fields(
        primary_mode=primary_mode,
        capabilities=capabilities,
        deliverables=deliverables,
        needs_artifact_context=needs_artifact_context,
    )
    confidence_band = _normalize_confidence_band(
        payload.get("confidence_band") or payload.get("confidence"),
        fallback_score=None,
    )
    confidence_score = _normalize_confidence_score(
        payload.get("confidence_score"),
        fallback_band=confidence_band,
    )
    confidence_band = _normalize_confidence_band(
        payload.get("confidence_band") or payload.get("confidence"),
        fallback_score=confidence_score,
    )
    ambiguity_flags = _normalize_conflict_flags(payload.get("ambiguity_flags") or payload.get("conflict_flags"))
    execution_plan = _normalize_string_list(payload.get("execution_plan") or payload.get("suggested_plan"))
    try:
        return IntentInterpretationPayload.model_validate(
            {
                "primary_mode": primary_mode,
                "confidence_score": confidence_score,
                "confidence_band": confidence_band,
                "needs_dataset": bool(payload.get("needs_dataset")) if payload.get("needs_dataset") is not None else _derive_needs_dataset(primary_mode, capabilities),
                "needs_tool_execution": bool(payload.get("needs_tool_execution")) if payload.get("needs_tool_execution") is not None else _derive_needs_tool_execution(primary_mode, capabilities),
                "needs_artifact_context": needs_artifact_context,
                "requested_capabilities": capabilities,
                "ambiguity_flags": ambiguity_flags,
                "guardrail_actions": _normalize_string_list(payload.get("guardrail_actions")),
                "fallback_reasons": _normalize_string_list(payload.get("fallback_reasons")),
                "reasoning_summary": str(payload.get("reasoning_summary", "")).strip(),
                "execution_plan": execution_plan,
                "intent_type": payload.get("intent_type") or compatibility["intent_type"],
                "is_dataset_overview": bool(payload.get("is_dataset_overview")) or compatibility["is_dataset_overview"],
                "is_follow_up": bool(payload.get("is_follow_up")) or compatibility["is_follow_up"],
                "requires_ml": bool(payload.get("requires_ml")) or compatibility["requires_ml"],
                "requires_chart": bool(payload.get("requires_chart")) or compatibility["requires_chart"],
                "requires_python_analysis": bool(payload.get("requires_python_analysis")) or compatibility["requires_python_analysis"],
                "deliverables": list(dict.fromkeys(deliverables or compatibility["deliverables"])),
                "confidence": confidence_band,
                "conflict_flags": ambiguity_flags,
                "route_source": payload.get("route_source", "llm_primary"),
                "suggested_plan": execution_plan,
            }
        )
    except ValidationError as exc:
        logger.debug("intent planner payload validation failed: %s", exc)
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

    routing_raw = _invoke_model_json(
        model,
        message=message,
        dataset_columns=resolved_dataset_columns,
        prior_analysis_active=prior_analysis_active,
        system_prompt=ROUTING_SYSTEM_PROMPT,
    )
    if routing_raw is None:
        return None
    routing_raw.setdefault("route_source", "llm_primary")
    routing_raw.setdefault("ambiguity_flags", routing_raw.get("conflict_flags", []))
    return _coerce_payload(routing_raw)
