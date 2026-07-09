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
from src.settings import SETTINGS
from src.tools import (
    bind_current_dataset_id,
    ml_execute,
    stats_execute,
    python_inter,
)

logger = logging.getLogger(__name__)
load_dotenv(override=True)

# Model selection - deepseek-v4-flash supports reasoning_content (think mode) for showing thought chains
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

    if primary_mode == "dataset_overview":
        return "This is a dataset overview request. Prefer explaining based on current schema/profile directly. Avoid unnecessary tool loops."
    if primary_mode == "modeling":
        return (
            "This is a clear modeling request. Take minimal necessary steps. "
            "Only call `ml_execute` when training, evaluation, or feature importance is actually needed."
        )
    if primary_mode == "visualization":
        return "This is a visualization request. Focus on data grouping, aggregation, and presenting numeric results in table form."
    if primary_mode == "clarification":
        return "Insufficient information. Prioritize clarifying the user's missing filters, target columns, or expected output. Do not enter complex tool loops directly."
    if primary_mode == "direct_answer":
        return "This is a direct answer request requiring no complex tools. Respond concisely. Only enter analytical workflow when truly necessary."

    return (
        "Select minimal necessary tools based on routing_decision. "
        "Stats questions → stats_execute, exploratory analysis → python_inter, "
        "clear modeling requests → `ml_execute`."
    )


model = ChatDeepSeek(
    model=DEFAULT_DEEPSEEK_MODEL,
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com",
)

tools = [
    python_inter,
    stats_execute,
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

def is_dataset_required(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
) -> bool:
    # Deprecated: no longer used for routing. LLM handles dataset requirement.
    return False


def is_stats_intent(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
) -> bool:
    # Deprecated: no longer used for routing. LLM handles intent detection.
    return False


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
                "Use **bold** for emphasis, `code` for values, ## for sections, and | tables | for structured data. "
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

[TOOLS]
You have 3 tools:
1. **python_inter(py_code)** -- execute pandas/numpy analysis in a secure sandbox (use for custom aggregations, merges, row-level logic)
2. **stats_execute(action, ...)** -- statistical analysis (grouping, t-test, chi-square, correlation, describe). Prefer this over python_inter for standard stats.
3. **ml_execute(action, ...)** -- machine learning (train, predict, metrics, feature importance)

IMPORTANT: Charts/plots are NOT available. Present all results as data tables and text summaries.

[Sandbox Rules]
ALLOWED: `df` (read-only DataFrame), `data`, `stats`, `profile`, `ml` helpers; `pd` (pandas), `np` (numpy); `print()`, `len()`, `sorted()`, `min()`, `max()`, `sum()`, `round()`, `list()`, `dict()`, `str()`, `int()`, `float()`, `set()`, `tuple()`, `bool()`, `range()`, `enumerate()`, `zip()`, `type()`, `dir()`, `any()`, `all()`, `abs()`
ALLOWED on `df`: `df["col"]`, `df.col_name`, `.groupby()`, `.agg()`, `.assign()`, `.merge()`, `.sort_values()`, `.rename()`, `.reset_index()`, `.value_counts()`, `.dropna()`, `.fillna()`, `.loc[]`, `.iloc[]`, `.copy()`, `.astype()`
BLOCKED: `import`, `eval`, `exec`, `open`, `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `compile`, `__import__`, `breakpoint`, `input`
BLOCKED identifiers: `os`, `sys`, `subprocess`, `shutil`, `requests`, `socket`, `pathlib`

[Helper APIs — available inside python_inter]
- `data.head(n)`, `data.describe()`, `data.numeric_summary()`, `data.missing_summary()`
- `data.value_counts("col")`, `data.unique("col")`, `data.correlation()`
- `data.group_mean("group_col", "value_col")`, `data.group_sum("group_col", "value_col")`
- `data.filter_equals("col", value)`, `data.select(["col1", "col2"])`
- `stats.describe_numeric("col")`, `stats.describe_categorical("col")`
- `stats.group_summary(group_col="col", value_col="col", agg="sum|mean|count|std|min|max")`
- `stats.t_test("col", group_col="g", group_value="x")`, `stats.chi_square("c1", "c2")`, `stats.anova("value", "group")`
- `profile.schema()`, `profile.analysis_preprocess()`, `profile.model_prep_plan()`
- `ml.linear_regression_fit(target="y", features=[...])`, `ml.logistic_fit(...)`, `ml.metrics()`, `ml.feature_importance()`

[Examples — inside python_inter]
  # GroupBy + agg using pandas
  monthly = df.assign(ym=df["order_date"].dt.to_period("M")).groupby("ym")["total_amount"].sum().reset_index()
  print(monthly)

  # Group summary via stats helper (inside python_inter)
  result = stats.group_summary(group_col="region", value_col="total_amount", agg="sum")
  print(result)

  # Value counts
  print(data.value_counts("region"))

[Pitfalls]
- Prefer `stats_execute` for standard stats; use `python_inter` only when stats_execute cannot express the query.
- When a tool returns an error, adapt your approach — do NOT retry the same code.
- `df["col"]` works for reading, but `df["col"] = ...` is blocked (read-only).
- Charts/plots are NOT available — present results as tables and text.
- `pd` and `np` are available without import.

[Your Responsibilities]
1. Complete calculations before answering. If results are empty, state it clearly.
2. Default to English. Use Markdown: **bold**, `code`, ## headings, | tables |.
3. Lead with the conclusion, then key evidence.
4. Never fabricate data, APIs, or conclusions.

[Multi-turn Context]
- Inherit filters and targets from the session.
- If a follow-up is ambiguous, confirm the scope in one sentence.

[Current Route Hint]
- {route_hint}

[When No Dataset]
- General chat: answer directly.
- Data analysis request: ask them to upload a CSV.
"""


graph = create_agent(
    model=model,
    tools=tools,
    middleware=[dataset_context_middleware],
    context_schema=AgentContext,
)
