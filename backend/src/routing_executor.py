from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from fastapi import Request
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent import AgentContext, _format_dataset_context_summary
from src.data_manager import get_data_context_summary, get_dataset, get_dataset_recommended_prompts
from src.errors import AppError
from src.request_context import get_request_id, set_degradation_mode, set_failure_stage
from src.result_types import get_artifact_repository
from src.routing_models import RoutingDecision
from src.settings import SETTINGS
from src.sse import backend_image_url, build_streaming_response, extract_text_from_chunk, format_sse
from src.tools import bind_current_dataset_id, consume_current_image_event

logger = logging.getLogger(__name__)

ML_DIRECT_TOOL_NAMES = {"ml_execute"}


@dataclass(slots=True)
class ExecutorDependencies:
    build_event_payload: Callable[..., dict[str, object]]
    build_error_sse: Callable[..., str]
    build_done_sse: Callable[..., str]
    classify_stream_exception: Callable[[Exception], AppError]
    extract_model_text: Callable[[Any], str]
    get_intent_planner_model: Callable[[], Any | None]
    generate_general_chat_reply: Callable[[list[dict[str, Any]]], Awaitable[str]]


@dataclass(slots=True)
class StreamOutcome:
    buffered_text_chunks: list[str]
    saw_chart_image: bool = False
    saw_ml_tool_call: bool = False
    produced_ml_artifact_types: set[str] | None = None
    emitted_text_chunk_count: int = 0
    tool_error_payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.produced_ml_artifact_types is None:
            self.produced_ml_artifact_types = set()


@dataclass(slots=True)
class LoopGuardState:
    total_steps: int = 0
    repeated_tool_signature: str | None = None
    repeated_tool_steps: int = 0
    no_progress_steps: int = 0
    progress_since_last_tool: bool = False

    def register_tool_start(self, tool_signature: str | None) -> None:
        self.total_steps += 1
        if tool_signature and tool_signature == self.repeated_tool_signature:
            self.repeated_tool_steps += 1
        else:
            self.repeated_tool_signature = tool_signature
            self.repeated_tool_steps = 1
        self.progress_since_last_tool = False

    def register_text_progress(self) -> None:
        self.progress_since_last_tool = True

    def register_artifact_progress(self) -> None:
        self.progress_since_last_tool = True

    def register_tool_end(self) -> None:
        if self.progress_since_last_tool:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1


def resolve_final_branch(
    *,
    dataset_id: str | None,
    routing_decision: RoutingDecision,
    **kwargs: Any,
) -> str:
    if not dataset_id:
        return "general_chat"
    return routing_decision.primary_mode


def build_dataset_overview_reply(dataset: object, *, recommended_prompts: list[str]) -> str:
    def _format_column_list(columns: list[str], *, limit: int = 8) -> str:
        if not columns:
            return "None"
        visible_columns = columns[:limit]
        suffix = f" and {len(columns)} more cols" if len(columns) > limit else ""
        return ", ".join(visible_columns) + suffix

    columns = getattr(dataset, "columns", [])
    numeric_columns = [
        str(column.get("name"))
        for column in columns
        if isinstance(column, dict) and column.get("type") == "numerical" and column.get("name")
    ]
    categorical_columns = [
        str(column.get("name"))
        for column in columns
        if isinstance(column, dict) and column.get("type") != "numerical" and column.get("name")
    ]
    schema_profile = getattr(dataset, "schema_profile_artifact", {})
    warnings = schema_profile.get("warnings", []) if isinstance(schema_profile, dict) else []
    warning_lines = [str(item) for item in warnings[:3] if item]

    lines = [
        "This dataset has been loaded. Let me give you a quick overview:",
        "",
        f"- Filename: {getattr(dataset, 'original_filename', 'uploaded.csv')}",
        f"- Shape: {getattr(dataset, 'row_count', 0):,} rows x {getattr(dataset, 'column_count', 0):,} cols",
        f"- Analysis basis: {getattr(dataset, 'analysis_basis', 'raw_df')}",
        f"- Numeric fields: {_format_column_list(numeric_columns)}",
        f"- Categorical/date fields: {_format_column_list(categorical_columns)}",
    ]
    if warning_lines:
        lines.extend(["", "I also noticed a few data quality / field type hints:"])
        lines.extend(f"- {warning}" for warning in warning_lines)
    lines.extend(["", "You can click the suggested questions above, or start from these directions:"])
    lines.extend(f"- {question}" for question in recommended_prompts)
    return "\n".join(lines)


async def generate_dataset_overview_reply(
    dataset: object,
    *,
    recommended_prompts: list[str],
    dependencies: ExecutorDependencies,
) -> str:
    model = dependencies.get_intent_planner_model()
    fallback = build_dataset_overview_reply(dataset, recommended_prompts=recommended_prompts)
    if model is None:
        return fallback

    schema_profile = getattr(dataset, "schema_profile_artifact", {})
    payload = {
        "filename": getattr(dataset, "original_filename", "uploaded.csv"),
        "dataset_id": getattr(dataset, "dataset_id", None),
        "analysis_basis": getattr(dataset, "analysis_basis", "raw_df"),
        "row_count": getattr(dataset, "row_count", 0),
        "column_count": getattr(dataset, "column_count", 0),
        "columns": getattr(dataset, "columns", []),
        "schema_profile": {
            "columns": schema_profile.get("columns", []) if isinstance(schema_profile, dict) else [],
            "warnings": schema_profile.get("warnings", []) if isinstance(schema_profile, dict) else [],
        },
        "preprocessing_log": getattr(dataset, "preprocessing_log", []),
        "recommended_prompts": recommended_prompts,
    }
    messages = [
        SystemMessage(
            content=(
                "You are Data Agent's dataset overview assistant. "
                "Based on the given structured context, write a concise, natural, trustworthy dataset overview. "
                "Must cover: data size, field categories, notable warnings, suggestions on where to start. "
                "Do not fabricate any statistics not present in the input. "
                "Output plain text, not JSON."
            )
        ),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    try:
        response = await model.ainvoke(messages)
    except Exception:
        logger.debug("dataset overview LLM generation failed", exc_info=True)
        return fallback

    text = dependencies.extract_model_text(response).strip()
    return text or fallback


async def generate_dataset_direct_reply(
    dataset_id: str,
    messages: list[dict[str, Any]],
    *,
    clarification: bool,
    dependencies: ExecutorDependencies,
) -> str:
    dataset_summary = _format_dataset_context_summary(get_data_context_summary(dataset_id))
    helper_message = (
        "The current question is better suited for direct answering. Only reference the dataset summary below when necessary, "
        "Do not fabricate statistics that were not provided.\n"
        if not clarification
        else "Current information is insufficient to safely perform data analysis. Based on the dataset summary below, "
        "Clearly state what information is missing, and suggest minimal viable clarifying questions.\n"
    )
    contextual_messages = [
        {
            "type": "assistant",
            "content": helper_message + "\nDataset Summary:\n" + dataset_summary,
        },
        *messages,
    ]
    return await dependencies.generate_general_chat_reply(contextual_messages)


def _build_tool_signature(tool_name: str | None, tool_data: object) -> str | None:
    if not isinstance(tool_name, str) or not tool_name.strip():
        return None
    if not isinstance(tool_data, dict) or not tool_data:
        return tool_name.strip()
    interesting_keys = (
        "action",
        "artifact_type",
        "columns",
        "col_a",
        "col_b",
        "features",
        "fname",
        "group_a",
        "group_b",
        "group_by",
        "group_col",
        "input",
        "inputs",
        "model_artifact_id",
        "model_type",
        "positive_label",
        "py_code",
        "sort_by",
        "target",
        "top_k",
        "top_n",
        "value_col",
    )
    compact_data = {key: tool_data.get(key) for key in interesting_keys if key in tool_data}
    if not compact_data:
        compact_data = tool_data
    try:
        serialized = json.dumps(compact_data, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        serialized = str(compact_data)
    return f"{tool_name.strip()}:{serialized[:240]}"


def _get_latest_artifact_info(dataset_id: str) -> tuple[str | None, str | None]:
    latest_artifact = get_artifact_repository().get_latest(dataset_id)
    if not isinstance(latest_artifact, dict):
        return None, None
    artifact_id = latest_artifact.get("artifact_id")
    artifact_type = latest_artifact.get("artifact_type")
    normalized_artifact_id = artifact_id if isinstance(artifact_id, str) and artifact_id.strip() else None
    normalized_artifact_type = artifact_type if isinstance(artifact_type, str) and artifact_type.strip() else None
    return normalized_artifact_id, normalized_artifact_type


def _extract_structured_artifact_type(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except Exception:
        return None
    artifact_type = payload.get("artifact_type")
    return artifact_type if isinstance(artifact_type, str) else None


def _extract_tool_error_payload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("artifact_type") != "tool_error":
        return None
    error_code = payload.get("error_code")
    if not isinstance(error_code, str) or not error_code.strip():
        return None
    return payload


def _raise_tool_error(payload: dict[str, Any]) -> None:
    message = payload.get("message")
    error_code = payload.get("error_code")
    retryable = payload.get("retryable")
    stage = payload.get("stage")
    if not isinstance(error_code, str) or not error_code.strip():
        return
    raise AppError(
        error_code,
        str(message or "Tool execution failed. Please try again later."),
        500,
        retryable=bool(retryable) if isinstance(retryable, bool) else False,
        stage=str(stage) if isinstance(stage, str) and stage.strip() else "tool_execution",
    )


def _check_loop_guard(loop_guard: LoopGuardState) -> None:
    if loop_guard.total_steps > SETTINGS.agent_max_total_steps:
        raise AppError(
            "agent_recursion_limit",
            "Task is too complex, Execution steps have reached the limit. Please break it down into more specific sub-questions.",
            500,
            stage="agent_loop",
        )
    if loop_guard.repeated_tool_steps > SETTINGS.agent_max_same_tool_steps:
        raise AppError(
            "agent_recursion_limit",
            "The same analysis tool has been called too many times. System has stopped this loop. Please narrow scope and retry.",
            500,
            stage="agent_loop",
        )
    if loop_guard.no_progress_steps > SETTINGS.agent_max_no_progress_steps:
        raise AppError(
            "agent_recursion_limit",
            "Multiple consecutive steps without effective progress. System has stopped. Please rephrase with a more specific question.",
            500,
            stage="agent_loop",
        )


async def _stream_graph_attempt(
    request: Request,
    *,
    graph: Any,
    requirements: Any,
    dataset_id: str,
    messages_for_graph: list[dict[str, object]],
    recursion_limit: int,
    dependencies: ExecutorDependencies,
) -> tuple[list[str], StreamOutcome]:
    set_failure_stage("model_stream")
    runtime_context = AgentContext(dataset_id=dataset_id)
    last_seen_artifact_id, _ = _get_latest_artifact_info(dataset_id)
    outcome = StreamOutcome(buffered_text_chunks=[])
    emitted_events: list[str] = []
    loop_guard = LoopGuardState()

    async for event in graph.astream_events(
        {"messages": messages_for_graph},
        config={"configurable": {"dataset_id": dataset_id}, "recursion_limit": recursion_limit},
        context=runtime_context,
        version="v2",
    ):
        event_name = str(event.get("event") or "")
        data = event.get("data") or {}
        name = event.get("name")
        if event_name == "on_tool_start":
            loop_guard.register_tool_start(
                _build_tool_signature(
                    str(name) if isinstance(name, str) else None,
                    data,
                )
            )
            _check_loop_guard(loop_guard)
            if isinstance(name, str) and name in ML_DIRECT_TOOL_NAMES:
                outcome.saw_ml_tool_call = True

            emitted_events.append(
                format_sse(
                    "tool_start",
                    dependencies.build_event_payload({"tool_name": name}, dataset_id=dataset_id),
                )
            )
            continue

        if event_name == "on_chain_stream":
            continue

        if event_name == "on_chat_model_stream":
            chunk = data.get("chunk") if isinstance(data, dict) else None
            text = extract_text_from_chunk(chunk)
            if not text:
                continue
            tool_error_payload = _extract_tool_error_payload(text)
            if tool_error_payload is not None:
                outcome.tool_error_payload = tool_error_payload
                continue
            loop_guard.register_text_progress()
            outcome.emitted_text_chunk_count += 1
            emitted_events.append(
                format_sse(
                    "message_chunk",
                    dependencies.build_event_payload({"content": text}, dataset_id=dataset_id),
                )
            )
            continue

        if event_name == "on_tool_end":
            if outcome.tool_error_payload is not None:
                _raise_tool_error(outcome.tool_error_payload)

            current_artifact_id, _ = _get_latest_artifact_info(dataset_id)
            if current_artifact_id and current_artifact_id != last_seen_artifact_id:
                loop_guard.register_artifact_progress()
                last_seen_artifact_id = current_artifact_id

            emitted_events.append(
                format_sse(
                    "tool_end",
                    dependencies.build_event_payload({"tool_name": name}, dataset_id=dataset_id),
                )
            )
            image_event = consume_current_image_event()
            if image_event:
                filename = image_event.get("filename")
                if isinstance(filename, str) and filename:
                    outcome.saw_chart_image = True
                    loop_guard.register_artifact_progress()
                    emitted_events.append(
                        format_sse(
                            "image_generated",
                            dependencies.build_event_payload(
                                {
                                    "type": "image_generated",
                                    "filename": filename,
                                    "image_url": backend_image_url(request, filename),
                                    "tool_name": image_event.get("tool_name"),
                                },
                                dataset_id=dataset_id,
                            ),
                        )
                    )
            loop_guard.register_tool_end()
            _check_loop_guard(loop_guard)

    return emitted_events, outcome


def execute_chat_stream_response(
    request: Request,
    *,
    graph: Any,
    requirements: Any,
    dependencies: ExecutorDependencies,
) -> Any:
    dataset_id = requirements.dataset_id
    route_info_payload = {
        "primary_mode": requirements.routing_decision.primary_mode,
        "needs_dataset": bool(dataset_id),
        "reasoning_summary": getattr(requirements.routing_decision, "reasoning_summary", None),
        "execution_plan": getattr(requirements.routing_decision, "execution_plan", None),
    }

    if not dataset_id:

        async def general_chat_event_generator():
            attempts = [
                ("none", requirements.compressed_messages),
                ("retry_once", requirements.compressed_messages),
            ]
            yield format_sse("route_info", dependencies.build_event_payload(route_info_payload))
            for index, (mode, messages_for_reply) in enumerate(attempts):
                try:
                    set_degradation_mode(mode)
                    set_failure_stage("model_stream")
                    reply = await dependencies.generate_general_chat_reply(messages_for_reply)
                    if reply.strip():
                        yield format_sse("message_chunk", dependencies.build_event_payload({"content": reply}))
                    yield dependencies.build_done_sse()
                    return
                except Exception as exc:
                    mapped_error = dependencies.classify_stream_exception(exc)
                    logger.exception(
                        "general chat stream failed",
                        extra={
                            "request_id": get_request_id(),
                            "attempt": index,
                            "degradation_mode": mode,
                            "error_code": mapped_error.code,
                            "failure_stage": mapped_error.stage,
                            "primary_mode": requirements.routing_decision.primary_mode,
                            "route_source": getattr(requirements.routing_decision, "route_source", None),
                        },
                    )
                    if mapped_error.retryable and index < len(attempts) - 1:
                        continue
                    yield dependencies.build_error_sse(mapped_error)
                    yield dependencies.build_done_sse()
                    return

        return build_streaming_response(general_chat_event_generator())

    dataset = get_dataset(dataset_id)
    if requirements.final_branch == "dataset_overview":

        async def dataset_overview_event_generator():
            with bind_current_dataset_id(dataset_id):
                recommended_prompts = get_dataset_recommended_prompts(dataset_id)
                yield format_sse(
                    "route_info",
                    dependencies.build_event_payload(route_info_payload, dataset_id=dataset_id),
                )
                yield format_sse(
                    "message_chunk",
                    dependencies.build_event_payload(
                        {
                            "content": await generate_dataset_overview_reply(
                                dataset,
                                recommended_prompts=recommended_prompts,
                                dependencies=dependencies,
                            )
                        },
                        dataset_id=dataset_id,
                    ),
                )
                yield dependencies.build_done_sse(dataset_id=dataset_id)

        return build_streaming_response(dataset_overview_event_generator())

    if requirements.final_branch in {"direct_answer", "clarification"}:

        async def dataset_direct_event_generator():
            with bind_current_dataset_id(dataset_id):
                attempts = [
                    ("none", requirements.compressed_messages),
                    ("retry_once", requirements.compressed_messages),
                ]
                yield format_sse(
                    "route_info",
                    dependencies.build_event_payload(route_info_payload, dataset_id=dataset_id),
                )
                for index, (mode, messages_for_reply) in enumerate(attempts):
                    try:
                        set_degradation_mode(mode)
                        set_failure_stage("model_stream")
                        reply = await generate_dataset_direct_reply(
                            dataset_id,
                            messages_for_reply,
                            clarification=requirements.final_branch == "clarification",
                            dependencies=dependencies,
                        )
                        if reply.strip():
                            yield format_sse(
                                "message_chunk",
                                dependencies.build_event_payload({"content": reply}, dataset_id=dataset_id),
                            )
                        yield dependencies.build_done_sse(dataset_id=dataset_id)
                        return
                    except Exception as exc:
                        mapped_error = dependencies.classify_stream_exception(exc)
                        logger.exception(
                            "dataset direct reply failed",
                            extra={
                                "request_id": get_request_id(),
                                "dataset_id": dataset_id,
                                "attempt": index,
                                "degradation_mode": mode,
                                "error_code": mapped_error.code,
                                "failure_stage": mapped_error.stage,
                                "primary_mode": requirements.routing_decision.primary_mode,
                                "route_source": getattr(requirements.routing_decision, "route_source", None),
                            },
                        )
                        if mapped_error.retryable and index < len(attempts) - 1:
                            continue
                        yield dependencies.build_error_sse(mapped_error, dataset_id=dataset_id)
                        yield dependencies.build_done_sse(dataset_id=dataset_id)
                        return

        return build_streaming_response(dataset_direct_event_generator())

    async def event_generator():
        with bind_current_dataset_id(dataset_id):
            attempts = [
                ("none", requirements.compressed_messages, SETTINGS.agent_default_recursion_limit),
                ("retry_once", requirements.compressed_messages, SETTINGS.agent_default_recursion_limit),
            ]
            try:
                yield format_sse(
                    "route_info",
                    dependencies.build_event_payload(route_info_payload, dataset_id=dataset_id),
                )
                for index, (mode, messages_for_graph, recursion_limit) in enumerate(attempts):
                    try:
                        set_degradation_mode(mode)
                        events, _ = await _stream_graph_attempt(
                            request,
                            graph=graph,
                            requirements=requirements,
                            dataset_id=dataset_id,
                            messages_for_graph=messages_for_graph,
                            recursion_limit=recursion_limit,
                            dependencies=dependencies,
                        )
                        for event in events:
                            yield event
                        yield dependencies.build_done_sse(dataset_id=dataset_id)
                        return
                    except Exception as exc:
                        mapped_error = dependencies.classify_stream_exception(exc)
                        set_failure_stage(mapped_error.stage)
                        logger.exception(
                            "chat stream failed",
                            extra={
                                "request_id": get_request_id(),
                                "dataset_id": dataset_id,
                                "degradation_mode": mode,
                                "history_message_count": requirements.history_message_count,
                                "compressed_history_count": requirements.compressed_history_count,
                                "artifact_refs_count": requirements.artifact_refs_count,
                                "failure_stage": mapped_error.stage,
                                "error_code": mapped_error.code,
                                "primary_mode": requirements.routing_decision.primary_mode,
                                "route_source": getattr(requirements.routing_decision, "route_source", None),
                                "attempt": index,
                            },
                        )
                        if mapped_error.retryable and index < len(attempts) - 1:
                            continue
                        if mapped_error.code == "agent_recursion_limit" and index < len(attempts) - 1:
                            continue
                        yield dependencies.build_error_sse(mapped_error, dataset_id=dataset_id)
                        yield dependencies.build_done_sse(dataset_id=dataset_id)
                        return
            except AppError as exc:
                set_failure_stage(exc.stage or "unknown")
                yield dependencies.build_error_sse(exc, dataset_id=dataset_id)
                yield dependencies.build_done_sse(dataset_id=dataset_id)
            except Exception as exc:
                mapped_error = dependencies.classify_stream_exception(exc)
                set_failure_stage(mapped_error.stage)
                logger.exception(
                    "chat stream failed",
                    extra={
                        "request_id": get_request_id(),
                        "dataset_id": dataset_id,
                        "failure_stage": mapped_error.stage,
                        "error_code": mapped_error.code,
                        "primary_mode": requirements.routing_decision.primary_mode,
                        "route_source": getattr(requirements.routing_decision, "route_source", None),
                    },
                )
                yield dependencies.build_error_sse(mapped_error, dataset_id=dataset_id)
                yield dependencies.build_done_sse(dataset_id=dataset_id)

    return build_streaming_response(event_generator())
