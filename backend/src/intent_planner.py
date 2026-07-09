from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from src.routing_models import RoutingDecision

logger = logging.getLogger(__name__)

INTENT_PLANNER_MODEL: Any | None = None
DEFAULT_INTENT_PLANNER_MODEL = os.getenv(
    "INTENT_PLANNER_MODEL",
    os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
)

ROUTING_SYSTEM_PROMPT = (
    'You are a routing agent. Analyze the user message and output ONLY valid JSON. '
    'No markdown, no explanation text, no extra prefixes or suffixes.\n'
    'Fields:\n'
    '- "primary_mode": one of "direct_answer", "dataset_overview", '
    '"analysis", "visualization", "modeling", "clarification"\n'
    '  • direct_answer: general Q&A, greetings, thanks, no data/tools needed\n'
    '  • dataset_overview: user wants an explanation of the current dataset\n'
    '  • analysis: statistical analysis, grouping, comparison, filtering, exploration\n'
    '  • visualization: charts, plots, graphs\n'
    '  • modeling: train a model, predict, evaluate, feature importance\n'
    '  • clarification: cannot determine what the user wants\n'
    '- "needs_dataset": true/false — does this require a loaded dataset?\n'
    '- "needs_tool_execution": true/false — does this require code execution or statistical tools?\n'
    '- "reasoning_summary": short one-sentence explanation\n'
    '- "execution_plan": array of strings describing next steps\n'
    'Output ONLY valid JSON.'
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


def _coerce_payload(payload: dict[str, Any]) -> RoutingDecision | None:
    try:
        execution_plan_raw = payload.get("execution_plan") or payload.get("suggested_plan") or []
        execution_plan = [str(s).strip() for s in execution_plan_raw if str(s).strip()]
        return RoutingDecision(
            primary_mode=str(payload.get("primary_mode", "analysis")).strip().lower().replace(" ", "_"),
            needs_dataset=bool(payload.get("needs_dataset", False)),
            needs_tool_execution=bool(payload.get("needs_tool_execution", False)),
            reasoning_summary=str(payload.get("reasoning_summary", "")).strip(),
            execution_plan=execution_plan,
            route_source=str(payload.get("route_source", "llm")).strip(),
        )
    except Exception:
        logger.debug("intent planner payload coercion failed", exc_info=True)
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
    except Exception as exc:
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


def plan_request_with_llm(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
    dataset_summary: dict[str, Any] | None = None,
    schema_profile: dict[str, Any] | None = None,
    latest_artifact: dict[str, Any] | None = None,
    available_artifact_types: list[str] | None = None,
    recommended_prompts: list[str] | None = None,
) -> RoutingDecision | None:
    model = get_intent_planner_model()
    if model is None:
        return None

    extra_payload: dict[str, Any] = {
        "dataset_summary": dataset_summary or {},
        "schema_profile": schema_profile or {},
        "latest_artifact": latest_artifact or {},
        "available_artifact_types": available_artifact_types or [],
        "recommended_prompts": recommended_prompts or [],
    }
    raw = _invoke_model_json(
        model,
        message=message,
        dataset_columns=dataset_columns or [],
        prior_analysis_active=prior_analysis_active,
        system_prompt=ROUTING_SYSTEM_PROMPT,
        extra_payload=extra_payload,
    )
    if raw is None:
        return None
    raw.setdefault("route_source", "llm")
    return _coerce_payload(raw)
