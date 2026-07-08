from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.data_manager import get_data_context_summary, get_dataset
from src.routing_rules import (
    RoutingContext,
    decide_dataset_required,
    decide_ml_intent,
    decide_stats_intent,
)
from src.settings import SETTINGS
from src.tools import (
    bind_current_dataset_id,
    fig_inter,
    ml_execute,
    stats_execute,
    python_inter,
)

logger = logging.getLogger(__name__)
load_dotenv(override=True)

# Use the non-thinking model by default to avoid DeepSeek reasoning_content
# round-trip errors in LangGraph agent tool loops.
DEFAULT_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


class AgentContext(BaseModel):
    dataset_id: str | None = Field(default=None, description="The current dataset ID being analyzed")
    routing_decision: dict[str, Any] | None = Field(
        default=None,
        description="Pre-generated routing decision from chat_service",
    )


def _extract_dataset_id_from_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, AgentContext):
        if value.dataset_id and value.dataset_id.strip():
            return value.dataset_id.strip()
        return None
    if isinstance(value, dict):
        direct_dataset_id = value.get("dataset_id")
        if isinstance(direct_dataset_id, str) and direct_dataset_id.strip():
            return direct_dataset_id.strip()
        configurable = value.get("configurable")
        if isinstance(configurable, dict):
            nested_dataset_id = configurable.get("dataset_id")
            if isinstance(nested_dataset_id, str) and nested_dataset_id.strip():
                return nested_dataset_id.strip()
        return None
    direct_dataset_id = getattr(value, "dataset_id", None)
    if isinstance(direct_dataset_id, str) and direct_dataset_id.strip():
        return direct_dataset_id.strip()
    configurable = getattr(value, "configurable", None)
    if isinstance(configurable, dict):
        nested_dataset_id = configurable.get("dataset_id")
        if isinstance(nested_dataset_id, str) and nested_dataset_id.strip():
            return nested_dataset_id.strip()
    return None


def _extract_dataset_id(request) -> str | None:
    runtime = getattr(request, "runtime", None)
    if runtime is not None:
        context = getattr(runtime, "context", None)
        dataset_id = _extract_dataset_id_from_value(context)
        if dataset_id:
            logger.debug(
                "dataset_id resolved from runtime.context",
                extra={"dataset_id": dataset_id},
            )
            return dataset_id

        runtime_config = getattr(runtime, "config", None)
        dataset_id = _extract_dataset_id_from_value(runtime_config)
        if dataset_id:
            logger.debug(
                "dataset_id resolved from runtime.config",
                extra={"dataset_id": dataset_id},
            )
            return dataset_id

    request_config = getattr(request, "config", None)
    dataset_id = _extract_dataset_id_from_value(request_config)
    if dataset_id:
        logger.debug(
            "dataset_id resolved from request.config",
            extra={"dataset_id": dataset_id},
        )
        return dataset_id
    return None


def _normalize_dict_like(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return None


def _extract_routing_decision_from_value(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, AgentContext):
        return _normalize_dict_like(value.routing_decision)
    if isinstance(value, dict):
        routing_decision = value.get("routing_decision")
        if routing_decision is not None:
            normalized = _normalize_dict_like(routing_decision)
            if normalized is not None:
                return normalized
        configurable = value.get("configurable")
        if isinstance(configurable, dict):
            nested = configurable.get("routing_decision")
            normalized = _normalize_dict_like(nested)
            if normalized is not None:
                return normalized
        return None
    direct_routing_decision = getattr(value, "routing_decision", None)
    normalized = _normalize_dict_like(direct_routing_decision)
    if normalized is not None:
        return normalized
    configurable = getattr(value, "configurable", None)
    if isinstance(configurable, dict):
        nested = configurable.get("routing_decision")
        normalized = _normalize_dict_like(nested)
        if normalized is not None:
            return normalized
    return None


def _build_route_hint(routing_decision: dict[str, Any] | None) -> str:
    if not routing_decision:
        return "No pre-injected routing_decision. Proceed with minimal necessary steps."

    primary_mode = str(routing_decision.get("primary_mode", "")).strip()
    requested_capabilities = {
        str(capability).strip()
        for capability in routing_decision.get("requested_capabilities", [])
        if isinstance(capability, str) and capability.strip()
    }

    if primary_mode == "dataset_overview":
        return "This is a dataset overview request. Prefer explaining based on current schema/profile directly. Avoid unnecessary tool loops."
    if primary_mode == "modeling":
        return (
            "This is a clear modeling request. Take minimal necessary steps. "
            "Only call `ml_execute` when training, evaluation, or feature importance is actually needed."
        )
    if primary_mode == "mixed":
        return (
            "This is a mixed workflow. Execute the analysis part first, then decide if `ml_execute` is needed. "
            "Do not escalate exploratory analysis into modeling."
        )
    if primary_mode == "visualization":
        return "This is a visualization request. Do minimal analysis first, then use `fig_inter` to generate the chart."
    if primary_mode == "artifact_followup":
        return "This is a follow-up request. Prefer reusing recent structured artifacts. Supplement with analysis or ML only if needed."
    if primary_mode == "clarification":
        return "Insufficient information. Prioritize clarifying the user's missing filters, target columns, or expected output. Do not enter complex tool loops directly."
    if primary_mode == "direct_answer":
        return "This is a direct answer request requiring no complex tools. Respond concisely. Only enter analytical workflow when truly necessary."

    if requested_capabilities.intersection({"stat_test", "group_analysis", "python_analysis"}):
        return "Statistical/exploratory intent detected. Prefer stats_execute or python_inter. Avoid unnecessary modeling."

    return (
        "Select minimal necessary tools based on routing_decision. "
        "Stats questions → stats_execute, exploratory analysis → python_inter, "
        "clear modeling requests → `ml_execute`, visualization → `fig_inter`."


model = ChatDeepSeek(
    model=DEFAULT_DEEPSEEK_MODEL,
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com",
)

tools = [
    python_inter,
    stats_execute,
    fig_inter,
    ml_execute,
]


def _format_dataset_context_summary(summary: dict[str, Any]) -> str:
    def _format_columns(columns: list[str], limit: int) -> str:
        if not columns:
            return "None"
        visible = columns[:limit]
        suffix = f" and {len(columns)} more columns" if len(columns) > limit else ""
        return ", ".join(visible) + suffix

    numeric_columns = [
        str(column)
        for column in summary.get("numeric_columns", [])
        if isinstance(column, str)
    ]
    categorical_columns = [
        str(column)
        for column in summary.get("categorical_columns", [])
        if isinstance(column, str)
    ]
    warnings = [
        str(item)
        for item in summary.get("warnings", [])
        if isinstance(item, str) and item.strip()
    ]
    preprocessing_log = [
        str(item)
        for item in summary.get("preprocessing_log", [])
        if isinstance(item, str) and item.strip()
    ]

    return (
        f"Filename: {summary.get('filename', 'uploaded.csv')}\n"
        f"dataset_id: {summary.get('dataset_id')}\n"
        f"Analysis basis: {summary.get('analysis_basis', 'raw_df')}\n"
        f"Shape: {summary.get('row_count', 0)} rows × {summary.get('column_count', 0)} cols\n"
        f"Numeric fields ({summary.get('numeric_column_count', 0)}): "
        f"{_format_columns(numeric_columns, SETTINGS.chat_max_prompt_numeric_columns)}\n"
        f"Categorical/date fields ({summary.get('categorical_column_count', 0)}): "
        f"{_format_columns(categorical_columns, SETTINGS.chat_max_prompt_categorical_columns)}\n"
        f"Preprocessing: {_format_columns(preprocessing_log, 3)}\n"
        f"Key warnings: {_format_columns(warnings, 3)}"
    )

def get_dataset_required_decision(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
):
    return decide_dataset_required(
        RoutingContext(
            message=message,
            dataset_columns=dataset_columns or [],
            prior_analysis_active=prior_analysis_active,
        )
    )


def is_dataset_required(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
) -> bool:
    decision = get_dataset_required_decision(
        message,
        dataset_columns=dataset_columns,
        prior_analysis_active=prior_analysis_active,
    )
    return decision.matched


def get_stats_intent_decision(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
):
    return decide_stats_intent(
        RoutingContext(
            message=message,
            dataset_columns=dataset_columns or [],
            prior_analysis_active=prior_analysis_active,
        )
    )


def is_stats_intent(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
) -> bool:
    decision = get_stats_intent_decision(
        message,
        dataset_columns=dataset_columns,
        prior_analysis_active=prior_analysis_active,
    )
    return decision.matched


def get_ml_intent_decision(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
):
    return decide_ml_intent(
        RoutingContext(
            message=message,
            dataset_columns=dataset_columns or [],
            prior_analysis_active=prior_analysis_active,
        )
    )


def is_ml_intent(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
) -> bool:
    decision = get_ml_intent_decision(
        message,
        dataset_columns=dataset_columns,
        prior_analysis_active=prior_analysis_active,
    )
    return decision.matched


def _build_general_chat_messages(messages: list[dict[str, Any]]) -> list[Any]:
    converted: list[Any] = [
        SystemMessage(
            content=(
                "You are Data Agent, an AI data analysis assistant. Your goal: accurate, concise, actionable. "
                "Default to English. Be professional and friendly. Never fabricate data, APIs, or conclusions. "
                "If the user asks general chat, concept explanations, study advice, or common knowledge, answer directly. "
                "If the user explicitly requests analysis based on uploaded data but no dataset is available, clearly ask them to upload a CSV first. "
                "When information is insufficient to draw a conclusion, state what's missing and suggest the minimal next step. "
            )
        )
    ]

    for message in messages:
        role = message.get("type")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role in {"human", "user"}:
            converted.append(HumanMessage(content=content))
        elif role in {"ai", "assistant"}:
            converted.append(AIMessage(content=content))

    return converted


def _extract_message_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            item_content = getattr(item, "text", None)
            if isinstance(item_content, str):
                parts.append(item_content)
            else:
                value = getattr(item, "content", None)
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return str(content)


async def generate_general_chat_reply(messages: list[dict[str, Any]]) -> str:
    response = await model.ainvoke(_build_general_chat_messages(messages))
    return _extract_message_text(response)


@dynamic_prompt
def dataset_context_middleware(request) -> str:
    dataset_id = _extract_dataset_id(request)
    routing_decision = None
    runtime = getattr(request, "runtime", None)
    if runtime is not None:
        routing_decision = _extract_routing_decision_from_value(getattr(runtime, "context", None))
    with bind_current_dataset_id(dataset_id):
        if dataset_id:
            dataset = get_dataset(dataset_id)
            data_context = _format_dataset_context_summary(get_data_context_summary(dataset_id))
            dataset_scope = f"Active dataset dataset_id: {dataset_id}, analysis based on {dataset.analysis_basis}."
        else:
            data_context = "No dataset selected. General chat can proceed; if data analysis is needed, please upload a CSV first."
            dataset_scope = "No active dataset. General chat can proceed directly. Data analysis requires uploading a CSV."

        route_hint = _build_route_hint(routing_decision)
        logger.debug("routing decision: %s", routing_decision)

        return f"""You are Data Agent, a senior data analysis assistant. Your primary goals: correct results, reviewable process, clear communication.

[Active Dataset]
{dataset_scope}

[Dataset Summary]
{data_context}

[Your Responsibilities]
1. For data-based questions, prioritize tools: data understanding/preprocessing -> `profile.*`, statistical analysis -> `stats.*`, clear modeling requests -> `ml_execute`, charts -> `fig_inter`.
2. Variable `df` is a read-only data view; `data`, `viz`, `stats`, `profile`, `ml` are whitelisted helper APIs. Prefer helper APIs; do not rely on undeclared capabilities.
   For clear modeling requests, first verify whether training/evaluation is truly needed before calling `ml_execute`. Do not mistake analytical requests for modeling requests.
   For statistical analysis, filtering, aggregation, comparison, and overviews, prefer `stats_execute`. Only use `python_inter` for more flexible exploratory analysis or plotting logic.
3. Answer only based on the current dataset and tool outputs. Never guess missing data or fabricate computed results.
4. Complete calculations before answering. If results are empty or samples insufficient, state this clearly and provide actionable next steps.
5. For charting tasks, provide a brief conclusion (what the chart shows) and keep titles/axis labels clear.
6. If the request involves non-existent columns, invalid filters, or unsupported operations, explain why and offer alternatives.
7. If the user asks about field semantics, modelable columns, or preprocessing steps, prioritize returning structured artifact results (e.g., schema_profile / preprocess_result / model_prep_plan) followed by a brief summary.
8. Baseline ML only supports logistic regression / linear regression. AutoML, Random Forest, XGBoost, SHAP are not supported. For out-of-scope requests, clearly refuse and suggest alternatives.
9. If the user explicitly requests model metrics or feature importance, continue calling `ml_execute` with `action="metrics"` / `action="feature_importance"`, reusing the latest model artifact.odel artifact.

[Output Style]
- Default to English. Lead with the conclusion, then provide key evidence (key numbers, group results, trends).
- Be concise. Do not repeat the user question or output meaningless template phrases.
- If the user requests Top N, filtering, grouping, or time aggregation, reflect whether these constraints were correctly applied.

[Multi-turn Context Rules]
- Inherit filters and target metrics from the current session (e.g., "now looking at California only").
- If a follow-up question is ambiguous, confirm the scope in one sentence before providing results.

[Current Route Hint]
- {route_hint}

[Rules When No Dataset]
- If it is general chat: answer directly.
- If the user explicitly wants data analysis: prompt them to upload a CSV first.
"""


graph = create_agent(
    model=model,
    tools=tools,
    middleware=[dataset_context_middleware],
    context_schema=AgentContext,
)
