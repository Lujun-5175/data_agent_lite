from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Request
from langgraph.errors import GraphRecursionError

from src.agent import generate_general_chat_reply, get_dataset_required_decision
from src.audit_log import get_audit_logger
from src.conversation_context import compress_conversation_messages
from src.data_manager import get_dataset
from src.errors import AppError
from src.intent_planner import (
    get_intent_planner_model,
    plan_request_with_llm,
    validate_task_plan_against_routing,
)
from src.request_context import (
    get_degradation_mode,
    get_request_id,
    set_route_diagnostics,
)
from src.request_parsing import (
    extract_dataset_id_from_payload,
    extract_latest_user_message,
    extract_messages,
    has_prior_analysis_context,
)
from src.result_types import get_artifact_repository
from src.plan_executor import build_executable_task_plan
from src.routing_executor import ExecutorDependencies, execute_chat_stream_response, resolve_final_branch
from src.routing_models import RoutingDecision
from src.routing_projection import (
    build_route_audit_payload,
    build_route_diagnostics,
    derive_legacy_route_projection,
)
from src.routing_signals import (
    collect_required_ml_artifacts,
    is_chart_request,
    is_explicit_ml_request,
    is_follow_up_request,
)
from src.routing_rules import RoutingContext, interpret_request_decision_from_llm
from src.sse import format_sse, now_iso
from src.task_plan_models import TaskPlan

logger = logging.getLogger(__name__)

ML_RESULT_ARTIFACT_TYPES = {"model_result", "metrics_result", "feature_importance_result"}


@dataclass(slots=True)
class ChatRequestRequirements:
    dataset_id: str | None
    messages: list[dict[str, object]]
    compressed_messages: list[dict[str, str]]
    simple_mode_messages: list[dict[str, str]]
    latest_user_message: str
    prior_analysis_active: bool
    routing_decision: RoutingDecision
    task_plan: TaskPlan | None
    task_plan_attempted: bool
    task_plan_generation_failed: bool
    chart_requested: bool
    explicit_ml_request: bool
    required_ml_artifacts: set[str]
    follow_up_message: dict[str, object] | None
    final_branch: str
    history_message_count: int
    compressed_history_count: int
    artifact_refs_count: int
    conversation_summary: dict[str, Any] | None

    @property
    def messages_for_graph(self) -> list[dict[str, object]]:
        messages_for_graph: list[dict[str, object]] = list(self.compressed_messages)
        if self.follow_up_message is not None:
            messages_for_graph.append(self.follow_up_message)
        return messages_for_graph

    @property
    def simple_messages_for_graph(self) -> list[dict[str, object]]:
        messages_for_graph: list[dict[str, object]] = list(self.simple_mode_messages)
        if self.follow_up_message is not None:
            messages_for_graph.append(self.follow_up_message)
        return messages_for_graph

    @property
    def legacy_route(self):
        return derive_legacy_route_projection(self.routing_decision)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _looks_like_follow_up_request(message: str) -> bool:
    return is_follow_up_request(_normalize_text(message))


def _looks_like_chart_request(message: str) -> bool:
    return is_chart_request(_normalize_text(message))


def _looks_like_explicit_ml_request(message: str) -> bool:
    return is_explicit_ml_request(_normalize_text(message))


def _collect_required_ml_artifacts(message: str) -> set[str]:
    return collect_required_ml_artifacts(_normalize_text(message))


def _build_follow_up_context_message(
    dataset_id: str,
    latest_user_message: str,
    *,
    force_follow_up: bool = False,
) -> dict[str, object] | None:
    if not force_follow_up and not _looks_like_follow_up_request(latest_user_message):
        return None

    latest_artifact = get_artifact_repository().get_latest(dataset_id)
    if not isinstance(latest_artifact, dict):
        return None

    artifact_type = str(latest_artifact.get("artifact_type", "unknown"))
    if artifact_type == "schema_profile":
        summary = {
            "artifact_type": artifact_type,
            "artifact_id": latest_artifact.get("artifact_id"),
            "dataset_id": latest_artifact.get("dataset_id"),
            "columns": latest_artifact.get("columns"),
            "warnings": latest_artifact.get("warnings", []),
        }
    elif artifact_type in ML_RESULT_ARTIFACT_TYPES:
        summary = {
            "artifact_type": artifact_type,
            "artifact_id": latest_artifact.get("artifact_id"),
            "dataset_id": latest_artifact.get("dataset_id"),
            "target": latest_artifact.get("target"),
            "model_type": latest_artifact.get("model_type"),
            "metrics": latest_artifact.get("metrics", {}),
            "items": latest_artifact.get("items", latest_artifact.get("coefficient_items", [])),
            "warnings": latest_artifact.get("warnings", []),
        }
    else:
        summary = {
            "artifact_type": artifact_type,
            "artifact_id": latest_artifact.get("artifact_id"),
            "dataset_id": latest_artifact.get("dataset_id"),
            "warnings": latest_artifact.get("warnings", []),
        }

    return {
        "type": "assistant",
        "content": "最近一次结构化结果，供解释或跟进使用：\n" + json.dumps(summary, ensure_ascii=False),
    }


def _build_event_payload(
    payload: dict[str, object] | None = None,
    *,
    dataset_id: str | None = None,
    include_done_metadata: bool = False,
) -> dict[str, object]:
    enriched = dict(payload or {})
    request_id = get_request_id()
    if request_id and "request_id" not in enriched:
        enriched["request_id"] = request_id
    if dataset_id and "dataset_id" not in enriched:
        enriched["dataset_id"] = dataset_id
    if "timestamp" not in enriched:
        enriched["timestamp"] = now_iso()
    if include_done_metadata:
        enriched["degradation_mode"] = get_degradation_mode()
    return enriched


def _build_error_sse(exc: AppError, *, dataset_id: str | None = None) -> str:
    return format_sse(
        "error",
        _build_event_payload(
            {
                "code": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
                "stage": exc.stage or "unknown",
            },
            dataset_id=dataset_id,
        ),
    )


def _build_done_sse(*, dataset_id: str | None = None) -> str:
    return format_sse("done", _build_event_payload({}, dataset_id=dataset_id, include_done_metadata=True))


def _extract_model_text(result: Any) -> str:
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


def _clean_exception_message(exc: Exception, *, limit: int = 240) -> str | None:
    text = " ".join(str(exc).strip().split())
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _classify_stream_exception(exc: Exception) -> AppError:
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, GraphRecursionError):
        return AppError(
            "agent_recursion_limit",
            "任务执行路径反复循环，已主动停止。请把问题拆成更具体的步骤后重试。",
            500,
            retryable=False,
            stage="agent_loop",
        )
    if isinstance(exc, httpx.ReadTimeout):
        return AppError(
            "upstream_model_timeout",
            "上游模型响应超时，请稍后重试。",
            500,
            retryable=True,
            stage="model_stream",
        )
    if isinstance(exc, httpx.TimeoutException):
        return AppError(
            "upstream_model_timeout",
            "上游模型响应超时，请稍后重试。",
            500,
            retryable=True,
            stage="model_stream",
        )
    if isinstance(exc, httpx.ReadError):
        return AppError(
            "upstream_model_stream_error",
            "上游模型流式响应中断，请稍后重试。",
            500,
            retryable=True,
            stage="model_stream",
        )
    if isinstance(exc, httpx.ConnectError):
        detail = _clean_exception_message(exc)
        message = "上游模型连接失败，请检查网络或模型服务配置。"
        if detail:
            message = f"{message} 详情：{detail}"
        return AppError(
            "upstream_model_connection_error",
            message,
            500,
            retryable=True,
            stage="model_stream",
        )
    if isinstance(exc, httpx.HTTPStatusError):
        detail = _clean_exception_message(exc)
        status_code = exc.response.status_code if exc.response is not None else None
        message = "上游模型服务返回异常状态。"
        if isinstance(status_code, int):
            message = f"{message} HTTP {status_code}。"
        if detail:
            message = f"{message} 详情：{detail}"
        return AppError(
            "upstream_model_http_error",
            message,
            500,
            retryable=bool(isinstance(status_code, int) and status_code >= 500),
            stage="model_stream",
        )
    if isinstance(exc, httpx.RequestError):
        detail = _clean_exception_message(exc)
        message = "上游模型请求失败，请检查网络连接后重试。"
        if detail:
            message = f"{message} 详情：{detail}"
        return AppError(
            "upstream_model_request_error",
            message,
            500,
            retryable=True,
            stage="model_stream",
        )
    detail = _clean_exception_message(exc)
    message = "服务器内部错误，请稍后重试。"
    if detail:
        message = f"模型调用失败：{detail}"
    return AppError(
        "internal_error",
        message,
        500,
        retryable=False,
        stage="model_stream",
    )


def _record_chat_route_audit(requirements: ChatRequestRequirements) -> None:
    try:
        get_audit_logger().record(
            tool_name="chat_route",
            dataset_id=requirements.dataset_id,
            tool_args={
                "intent_type": requirements.legacy_route.intent_type,
                "chart_requested": requirements.chart_requested,
                "explicit_ml_request": requirements.explicit_ml_request,
                "is_dataset_overview": requirements.final_branch == "dataset_overview",
            },
            execution_status="success",
            latency_ms=0.0,
            extra={
                "request_id": get_request_id(),
                "message_preview": requirements.latest_user_message[:120],
                "route_stage": "request_analysis",
                "routing": build_route_audit_payload(
                    requirements.routing_decision,
                    final_branch=requirements.final_branch,
                    task_plan=requirements.task_plan,
                    task_plan_attempted=requirements.task_plan_attempted,
                    task_plan_generation_failed=requirements.task_plan_generation_failed,
                ),
            },
        )
    except Exception:
        logger.warning("Audit logging failed for chat route", exc_info=True)


def analyze_chat_request(payload: dict[str, object]) -> ChatRequestRequirements:
    dataset_id = extract_dataset_id_from_payload(payload)
    messages = extract_messages(payload)
    if not messages:
        raise AppError("validation_error", "请求参数不合法，请先输入问题。", 422, stage="validation")

    latest_user_message = extract_latest_user_message(messages)
    prior_analysis_active = has_prior_analysis_context(messages)
    dataset_columns: list[str] = []
    latest_artifact: dict[str, object] | None = None
    dataset_summary: dict[str, Any] | None = None
    schema_profile: dict[str, Any] | None = None
    recommended_prompts: list[str] = []

    if not dataset_id:
        dataset_required_decision = get_dataset_required_decision(
            latest_user_message,
            dataset_columns=[],
            prior_analysis_active=prior_analysis_active,
        )
        if dataset_required_decision.matched:
            raise AppError(
                "dataset_required",
                "当前未选择数据集，请先上传 CSV 文件后再进行数据分析。",
                400,
                stage="validation",
            )
    else:
        dataset = get_dataset(dataset_id)
        latest_artifact = get_artifact_repository().get_latest(dataset_id)
        dataset_summary = {
            "dataset_id": dataset.dataset_id,
            "filename": dataset.original_filename,
            "analysis_basis": dataset.analysis_basis,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
            "columns": list(dataset.columns),
            "preprocessing_log": list(dataset.preprocessing_log),
        }
        schema_profile = (
            dict(dataset.schema_profile_artifact)
            if isinstance(dataset.schema_profile_artifact, dict)
            else None
        )
        recommended_prompts = list(getattr(dataset, "recommended_prompts", []) or [])
        dataset_columns = [
            str(column.get("name"))
            for column in getattr(dataset, "columns", [])
            if isinstance(column, dict) and column.get("name")
        ]

    compressed = compress_conversation_messages(messages, dataset_id=dataset_id)
    if not compressed.messages_for_model:
        raise AppError("history_compression_error", "会话历史压缩失败，请重新发起请求。", 422, stage="history_compression")

    routing_context = RoutingContext(
        message=latest_user_message,
        dataset_columns=dataset_columns,
        prior_analysis_active=prior_analysis_active,
        latest_artifact=latest_artifact,
    )
    latest_artifact_type = latest_artifact.get("artifact_type") if isinstance(latest_artifact, dict) else None
    available_artifact_types = [latest_artifact_type.strip()] if isinstance(latest_artifact_type, str) and latest_artifact_type.strip() else []
    planning_result = plan_request_with_llm(
        latest_user_message,
        dataset_columns=dataset_columns,
        prior_analysis_active=prior_analysis_active,
        dataset_summary=dataset_summary,
        schema_profile=schema_profile,
        latest_artifact=latest_artifact,
        available_artifact_types=available_artifact_types,
        recommended_prompts=recommended_prompts,
    )
    routing_decision = interpret_request_decision_from_llm(
        routing_context,
        llm_decision=planning_result.routing_decision if planning_result is not None else None,
    )
    legacy_route = derive_legacy_route_projection(routing_decision)
    task_plan_attempted = routing_decision.needs_dataset or routing_decision.needs_tool_execution
    task_plan = validate_task_plan_against_routing(
        routing_decision=routing_decision,
        task_plan=planning_result.task_plan if planning_result is not None else None,
    )
    task_plan = build_executable_task_plan(task_plan)
    final_branch = resolve_final_branch(
        dataset_id=dataset_id,
        routing_decision=routing_decision,
        task_plan=task_plan,
    )
    set_route_diagnostics(
        build_route_diagnostics(
            routing_decision,
            final_branch=final_branch,
            task_plan=task_plan,
            task_plan_attempted=task_plan_attempted,
            task_plan_generation_failed=task_plan_attempted and task_plan is None,
        )
    )
    simple_mode_messages = []
    if compressed.summary_message is not None:
        summary_message = compressed.summary_message
        latest_user_message_for_simple: dict[str, str] | None = None
        latest_assistant_message_for_simple: dict[str, str] | None = None
        for message in reversed(compressed.messages_for_model):
            role = str(message.get("type", "user"))
            if latest_user_message_for_simple is None and role in {"human", "user"}:
                latest_user_message_for_simple = message
                continue
            if (
                latest_assistant_message_for_simple is None
                and role in {"ai", "assistant"}
                and message is not summary_message
            ):
                latest_assistant_message_for_simple = message
            if latest_user_message_for_simple is not None and latest_assistant_message_for_simple is not None:
                break
        simple_mode_messages.append(summary_message)
        if latest_assistant_message_for_simple is not None:
            simple_mode_messages.append(latest_assistant_message_for_simple)
        if latest_user_message_for_simple is not None:
            simple_mode_messages.append(latest_user_message_for_simple)
    else:
        simple_mode_messages = list(compressed.messages_for_model[-2:])
    if not simple_mode_messages:
        simple_mode_messages = list(compressed.messages_for_model)

    return ChatRequestRequirements(
        dataset_id=dataset_id,
        messages=messages,
        compressed_messages=compressed.messages_for_model,
        simple_mode_messages=simple_mode_messages,
        latest_user_message=latest_user_message,
        prior_analysis_active=prior_analysis_active,
        routing_decision=routing_decision,
        task_plan=task_plan,
        task_plan_attempted=task_plan_attempted,
        task_plan_generation_failed=task_plan_attempted and task_plan is None,
        chart_requested=legacy_route.requires_chart or _looks_like_chart_request(latest_user_message),
        explicit_ml_request=_looks_like_explicit_ml_request(latest_user_message),
        required_ml_artifacts=_collect_required_ml_artifacts(latest_user_message),
        follow_up_message=(
            _build_follow_up_context_message(
                dataset_id,
                latest_user_message,
                force_follow_up=legacy_route.is_follow_up,
            )
            if dataset_id
            else None
        ),
        final_branch=final_branch,
        history_message_count=compressed.history_message_count,
        compressed_history_count=compressed.compressed_history_count,
        artifact_refs_count=compressed.artifact_refs_count,
        conversation_summary=compressed.conversation_summary,
    )


async def create_chat_stream_response(
    request: Request,
    payload: dict[str, object],
    *,
    graph: Any,
) -> Any:
    requirements = analyze_chat_request(payload)
    _record_chat_route_audit(requirements)

    logger.info(
        "chat_stream payload received",
        extra={
            "request_id": get_request_id(),
            "dataset_id": requirements.dataset_id,
            "message_preview": requirements.latest_user_message[:80],
            "intent_type": requirements.legacy_route.intent_type,
            "intent_confidence": requirements.legacy_route.confidence,
            "intent_conflict_flags": requirements.legacy_route.conflict_flags,
            "intent_route_source": requirements.legacy_route.route_source,
            "history_message_count": requirements.history_message_count,
            "compressed_history_count": requirements.compressed_history_count,
            "artifact_refs_count": requirements.artifact_refs_count,
            "degradation_mode": get_degradation_mode(),
        },
    )
    dependencies = ExecutorDependencies(
        build_event_payload=_build_event_payload,
        build_error_sse=_build_error_sse,
        build_done_sse=_build_done_sse,
        classify_stream_exception=_classify_stream_exception,
        extract_model_text=_extract_model_text,
        get_intent_planner_model=get_intent_planner_model,
        generate_general_chat_reply=generate_general_chat_reply,
    )
    return execute_chat_stream_response(
        request,
        graph=graph,
        requirements=requirements,
        dependencies=dependencies,
    )
