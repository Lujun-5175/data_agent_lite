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
from src.plan_executor import execute_task_plan, supports_task_plan
from src.plan_verifier import verify_task_plan
from src.request_context import get_request_id, set_degradation_mode, set_failure_stage
from src.result_types import get_artifact_repository
from src.routing_models import RoutingDecision
from src.routing_projection import build_route_info_payload
from src.settings import SETTINGS
from src.sse import backend_image_url, build_streaming_response, extract_text_from_chunk, format_sse
from src.tools import bind_current_dataset_id, consume_current_image_event

logger = logging.getLogger(__name__)

ML_RESULT_ARTIFACT_TYPES = {"model_result", "metrics_result", "feature_importance_result"}
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
    task_plan: Any | None = None,
) -> str:
    if not dataset_id:
        return "general_chat"
    if routing_decision.primary_mode == "dataset_overview":
        return "dataset_overview"
    if routing_decision.primary_mode == "clarification":
        return "clarification"
    if routing_decision.primary_mode == "direct_answer" and not routing_decision.needs_tool_execution:
        return "direct_answer"
    if supports_task_plan(task_plan):
        return "plan_execution"
    return "agent_graph"


def build_dataset_overview_reply(dataset: object, *, recommended_prompts: list[str]) -> str:
    def _format_column_list(columns: list[str], *, limit: int = 8) -> str:
        if not columns:
            return "暂无"
        visible_columns = columns[:limit]
        suffix = f" 等 {len(columns)} 列" if len(columns) > limit else ""
        return "、".join(visible_columns) + suffix

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
        "这份数据集已经加载好了，我先帮你快速讲解一下：",
        "",
        f"- 文件名：{getattr(dataset, 'original_filename', 'uploaded.csv')}",
        f"- 数据规模：{getattr(dataset, 'row_count', 0):,} 行 × {getattr(dataset, 'column_count', 0):,} 列",
        f"- 分析基准：{getattr(dataset, 'analysis_basis', 'raw_df')}",
        f"- 数值字段：{_format_column_list(numeric_columns)}",
        f"- 分类/日期字段：{_format_column_list(categorical_columns)}",
    ]
    if warning_lines:
        lines.extend(["", "我也注意到几个数据质量/字段类型提示："])
        lines.extend(f"- {warning}" for warning in warning_lines)
    lines.extend(["", "你可以直接点上方推荐问题，或者从这些方向开始："])
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
                "你是 Data Agent 的数据集概览助手。"
                "请基于给定的结构化上下文，用中文写一段简洁、自然、可信的数据集讲解。"
                "必须覆盖：数据规模、字段大类、值得注意的 warning、建议从哪些问题开始。"
                "不要编造任何未出现在输入中的统计值。"
                "输出纯文本，不要输出 JSON。"
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
        "当前用户问题更适合直接回答。请仅在必要时引用下面的数据集摘要，不要虚构未给出的统计值。\n"
        if not clarification
        else "当前信息不足以安全执行数据分析。请基于下面的数据集摘要，用中文明确指出缺失信息，并提出最小可行澄清问题。\n"
    )
    contextual_messages = [
        {
            "type": "assistant",
            "content": helper_message + "\n数据集摘要：\n" + dataset_summary,
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


def _looks_like_internal_intent_payload(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        payload = json.loads(stripped)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    intent_keys = {
        "intent_type",
        "is_dataset_overview",
        "is_follow_up",
        "requires_ml",
        "requires_chart",
        "requires_python_analysis",
        "confidence",
        "conflict_flags",
        "route_source",
        "reasoning_summary",
        "suggested_plan",
    }
    return "intent_type" in payload and len(intent_keys.intersection(payload)) >= 3


def _strip_internal_intent_payload_prefix(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return text

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[: index + 1]
                if _looks_like_internal_intent_payload(candidate):
                    return stripped[index + 1 :].lstrip()
                return text
    return text


def _raise_tool_error(payload: dict[str, Any]) -> None:
    message = payload.get("message")
    error_code = payload.get("error_code")
    retryable = payload.get("retryable")
    stage = payload.get("stage")
    if not isinstance(error_code, str) or not error_code.strip():
        return
    raise AppError(
        error_code,
        str(message or "工具执行失败，请稍后重试。"),
        500,
        retryable=bool(retryable) if isinstance(retryable, bool) else False,
        stage=str(stage) if isinstance(stage, str) and stage.strip() else "tool_execution",
    )


def _check_loop_guard(loop_guard: LoopGuardState) -> None:
    if loop_guard.total_steps > SETTINGS.agent_max_total_steps:
        raise AppError(
            "agent_recursion_limit",
            "任务过于复杂，执行步骤已达到上限。请拆成更具体的子问题后重试。",
            500,
            stage="agent_loop",
        )
    if loop_guard.repeated_tool_steps > SETTINGS.agent_max_same_tool_steps:
        raise AppError(
            "agent_recursion_limit",
            "同一分析工具被重复调用过多次，系统已停止本次循环。请缩小范围后重试。",
            500,
            stage="agent_loop",
        )
    if loop_guard.no_progress_steps > SETTINGS.agent_max_no_progress_steps:
        raise AppError(
            "agent_recursion_limit",
            "任务连续多步没有产生有效进展，系统已主动停止。请改成更具体的问题后重试。",
            500,
            stage="agent_loop",
        )


def _validate_stream_outcome(requirements: Any, outcome: StreamOutcome) -> None:
    if requirements.explicit_ml_request and not outcome.saw_ml_tool_call:
        raise AppError(
            "structured_failure",
            "本次建模请求没有调用直接的 ml 工具，请先通过 ml_execute 完成建模。",
            200,
            stage="validation",
        )
    if requirements.explicit_ml_request:
        missing_ml_artifacts = requirements.required_ml_artifacts - outcome.produced_ml_artifact_types
        if missing_ml_artifacts:
            raise AppError(
                "structured_failure",
                f"本次建模请求缺少结构化结果：{', '.join(sorted(missing_ml_artifacts))}。",
                200,
                stage="validation",
            )
    if requirements.chart_requested and not outcome.saw_chart_image:
        raise AppError(
            "structured_failure",
            "本次图表请求没有成功生成可展示的图片结果，请检查字段名或图表描述后重试。",
            200,
            stage="validation",
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
    runtime_context = AgentContext(
        dataset_id=dataset_id,
        routing_decision=requirements.routing_decision.model_dump(),
    )
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
            loop_guard.register_tool_start(_build_tool_signature(str(name) if isinstance(name, str) else None, data))
            _check_loop_guard(loop_guard)
            if isinstance(name, str) and name in ML_DIRECT_TOOL_NAMES:
                outcome.saw_ml_tool_call = True

        if event_name == "on_chain_stream":
            continue

        if event_name == "on_chat_model_stream":
            chunk = data.get("chunk") if isinstance(data, dict) else None
            text = _strip_internal_intent_payload_prefix(extract_text_from_chunk(chunk))
            if _looks_like_internal_intent_payload(text) or not text:
                continue
            tool_error_payload = _extract_tool_error_payload(text)
            if tool_error_payload is not None:
                outcome.tool_error_payload = tool_error_payload
                continue
            loop_guard.register_text_progress()
            outcome.emitted_text_chunk_count += 1
            if requirements.explicit_ml_request or requirements.chart_requested:
                outcome.buffered_text_chunks.append(text)
                artifact_type = _extract_structured_artifact_type(text)
                if artifact_type in ML_RESULT_ARTIFACT_TYPES:
                    current_artifact = get_artifact_repository().get_latest(dataset_id, artifact_type=artifact_type)
                    current_artifact_id = current_artifact.get("artifact_id") if current_artifact else None
                    if current_artifact_id and current_artifact_id != last_seen_artifact_id:
                        outcome.produced_ml_artifact_types.add(str(artifact_type))
                        loop_guard.register_artifact_progress()
            else:
                emitted_events.append(
                    format_sse(
                        "message_chunk",
                        dependencies.build_event_payload({"content": text}, dataset_id=dataset_id),
                    )
                )
            continue

        if event_name == "on_tool_start":
            emitted_events.append(
                format_sse(
                    "tool_start",
                    dependencies.build_event_payload({"tool_name": name}, dataset_id=dataset_id),
                )
            )
            continue

        if event_name == "on_tool_end":
            if outcome.tool_error_payload is not None:
                _raise_tool_error(outcome.tool_error_payload)
            current_artifact_id, current_artifact_type = _get_latest_artifact_info(dataset_id)
            if current_artifact_id and current_artifact_id != last_seen_artifact_id:
                loop_guard.register_artifact_progress()
                last_seen_artifact_id = current_artifact_id
                if current_artifact_type in ML_RESULT_ARTIFACT_TYPES:
                    outcome.produced_ml_artifact_types.add(str(current_artifact_type))
            elif isinstance(name, str) and name in ML_DIRECT_TOOL_NAMES and current_artifact_type in ML_RESULT_ARTIFACT_TYPES:
                outcome.produced_ml_artifact_types.add(str(current_artifact_type))

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

    _validate_stream_outcome(requirements, outcome)
    for text in outcome.buffered_text_chunks:
        emitted_events.append(
            format_sse(
                "message_chunk",
                dependencies.build_event_payload({"content": text}, dataset_id=dataset_id),
            )
        )
    return emitted_events, outcome


def execute_chat_stream_response(
    request: Request,
    *,
    graph: Any,
    requirements: Any,
    dependencies: ExecutorDependencies,
) -> Any:
    dataset_id = requirements.dataset_id
    route_info_payload = build_route_info_payload(
        requirements.routing_decision,
        final_branch=requirements.final_branch,
        task_plan=requirements.task_plan,
        task_plan_attempted=requirements.task_plan_attempted,
        task_plan_generation_failed=requirements.task_plan_generation_failed,
    )

    if not dataset_id:

        async def general_chat_event_generator():
            attempts = [
                ("none", requirements.compressed_messages),
                ("retry_stream_once", requirements.compressed_messages),
                ("non_stream_simple_mode", requirements.simple_mode_messages),
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
                            "intent_confidence": requirements.legacy_route.confidence,
                            "intent_route_source": requirements.legacy_route.route_source,
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
                    ("retry_stream_once", requirements.compressed_messages),
                    ("non_stream_simple_mode", requirements.simple_mode_messages),
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
                                "intent_route_source": requirements.legacy_route.route_source,
                            },
                        )
                        if mapped_error.retryable and index < len(attempts) - 1:
                            continue
                        yield dependencies.build_error_sse(mapped_error, dataset_id=dataset_id)
                        yield dependencies.build_done_sse(dataset_id=dataset_id)
                        return

        return build_streaming_response(dataset_direct_event_generator())

    if requirements.final_branch == "plan_execution":

        async def plan_execution_event_generator():
            with bind_current_dataset_id(dataset_id):
                yield format_sse(
                    "route_info",
                    dependencies.build_event_payload(route_info_payload, dataset_id=dataset_id),
                )
                try:
                    result = execute_task_plan(dataset_id=dataset_id, task_plan=requirements.task_plan)
                    verification = verify_task_plan(
                        task_plan=requirements.task_plan,
                        executed_task_ids=result.executed_task_ids,
                        produced_outputs=result.produced_outputs,
                    )
                    if verification.status != "success":
                        raise AppError(
                            "task_plan_incomplete",
                            verification.reason or "结构化计划未完整执行。",
                            200,
                            stage="plan_verification",
                        )
                    if result.content.strip():
                        yield format_sse(
                            "message_chunk",
                            dependencies.build_event_payload({"content": result.content}, dataset_id=dataset_id),
                        )
                    yield dependencies.build_done_sse(dataset_id=dataset_id)
                    return
                except AppError as exc:
                    if exc.code != "unsupported_task_plan":
                        yield dependencies.build_error_sse(exc, dataset_id=dataset_id)
                        yield dependencies.build_done_sse(dataset_id=dataset_id)
                        return

            async for event in event_generator():
                yield event

        async def event_generator():
            with bind_current_dataset_id(dataset_id):
                attempts = [
                    ("none", requirements.messages_for_graph, SETTINGS.agent_default_recursion_limit),
                    ("retry_stream_once", requirements.messages_for_graph, SETTINGS.agent_default_recursion_limit),
                    ("non_stream_simple_mode", requirements.simple_messages_for_graph, SETTINGS.agent_simple_mode_recursion_limit),
                ]
                try:
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
                                    "intent_type": requirements.legacy_route.intent_type,
                                    "intent_confidence": requirements.legacy_route.confidence,
                                    "intent_conflict_flags": requirements.legacy_route.conflict_flags,
                                    "intent_route_source": requirements.legacy_route.route_source,
                                    "failure_stage": mapped_error.stage,
                                    "error_code": mapped_error.code,
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
                            "intent_confidence": requirements.legacy_route.confidence,
                            "intent_route_source": requirements.legacy_route.route_source,
                        },
                    )
                    yield dependencies.build_error_sse(mapped_error, dataset_id=dataset_id)
                    yield dependencies.build_done_sse(dataset_id=dataset_id)

        return build_streaming_response(plan_execution_event_generator())

    async def event_generator():
        with bind_current_dataset_id(dataset_id):
            attempts = [
                ("none", requirements.messages_for_graph, SETTINGS.agent_default_recursion_limit),
                ("retry_stream_once", requirements.messages_for_graph, SETTINGS.agent_default_recursion_limit),
                ("non_stream_simple_mode", requirements.simple_messages_for_graph, SETTINGS.agent_simple_mode_recursion_limit),
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
                                "intent_type": requirements.legacy_route.intent_type,
                                "intent_confidence": requirements.legacy_route.confidence,
                                "intent_conflict_flags": requirements.legacy_route.conflict_flags,
                                "intent_route_source": requirements.legacy_route.route_source,
                                "failure_stage": mapped_error.stage,
                                "error_code": mapped_error.code,
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
                        "intent_confidence": requirements.legacy_route.confidence,
                        "intent_route_source": requirements.legacy_route.route_source,
                    },
                )
                yield dependencies.build_error_sse(mapped_error, dataset_id=dataset_id)
                yield dependencies.build_done_sse(dataset_id=dataset_id)

    return build_streaming_response(event_generator())
