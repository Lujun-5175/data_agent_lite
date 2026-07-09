from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Request
from langgraph.errors import GraphRecursionError

from src.agent import generate_general_chat_reply
from src.audit_log import get_audit_logger
from src.conversation_context import compress_conversation_messages
from src.data_manager import get_dataset
from src.errors import AppError
from src.intent_planner import (
    get_intent_planner_model,
    plan_request_with_llm,
)
from src.request_context import (
    get_degradation_mode,
    get_request_id,
)
from src.request_parsing import (
    extract_dataset_id_from_payload,
    extract_latest_user_message,
    extract_messages,
    has_prior_analysis_context,
)
from src.result_types import get_artifact_repository
from src.routing_executor import ExecutorDependencies, execute_chat_stream_response, resolve_final_branch
from src.routing_models import RoutingDecision
from src.sse import format_sse, now_iso

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class ChatRequestRequirements:
    dataset_id: str | None
    messages: list[dict[str, object]]
    compressed_messages: list[dict[str, str]]
    latest_user_message: str
    prior_analysis_active: bool
    routing_decision: RoutingDecision
    final_branch: str
    history_message_count: int
    compressed_history_count: int
    artifact_refs_count: int
    conversation_summary: dict[str, Any] | None


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
            "Task execution path looped repeatedly. System stopped. Please break your question into more specific steps.",
            500,
            retryable=False,
            stage="agent_loop",
        )
    if isinstance(exc, httpx.ReadTimeout):
        return AppError(
            "upstream_model_timeout",
            "Upstream model timed out. Please try again.",
            500,
            retryable=True,
            stage="model_stream",
        )
    if isinstance(exc, httpx.TimeoutException):
        return AppError(
            "upstream_model_timeout",
            "Upstream model timed out. Please try again.",
            500,
            retryable=True,
            stage="model_stream",
        )
    if isinstance(exc, httpx.ReadError):
        return AppError(
            "upstream_model_stream_error",
            "Upstream model stream was interrupted. Please try again.",
            500,
            retryable=True,
            stage="model_stream",
        )
    if isinstance(exc, httpx.ConnectError):
        detail = _clean_exception_message(exc)
        message = "Upstream model connection failed. Check network or model service configuration."
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
        message = "Upstream model service returned an abnormal status."
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
        message = "Upstream model request failed. Check network connection and try again."
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
    message = "Internal server error. Please try again later."
    if detail:
        message = f"Model call failed: {detail}"
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
                "primary_mode": requirements.routing_decision.primary_mode,
                "needs_dataset": requirements.routing_decision.needs_dataset,
            },
            execution_status="success",
            latency_ms=0.0,
            extra={
                "request_id": get_request_id(),
                "message_preview": requirements.latest_user_message[:120],
                "route_stage": "request_analysis",
                "routing": {
                    "primary_mode": requirements.routing_decision.primary_mode,
                    "needs_dataset": requirements.routing_decision.needs_dataset,
                    "final_branch": requirements.final_branch,
                    "route_source": requirements.routing_decision.route_source,
                },
            },
        )
    except Exception:
        logger.warning("Audit logging failed for chat route", exc_info=True)


def analyze_chat_request(payload: dict[str, object]) -> ChatRequestRequirements:
    dataset_id = extract_dataset_id_from_payload(payload)
    messages = extract_messages(payload)
    if not messages:
        raise AppError("validation_error", "Invalid request parameters. Please enter a question first.", 422, stage="validation")

    latest_user_message = extract_latest_user_message(messages)
    prior_analysis_active = has_prior_analysis_context(messages)
    dataset_columns: list[str] = []
    latest_artifact: dict[str, object] | None = None
    dataset_summary: dict[str, Any] | None = None
    schema_profile: dict[str, Any] | None = None
    recommended_prompts: list[str] = []

    if dataset_id:
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
        raise AppError("history_compression_error", "Conversation history compression failed. Please send a new request.", 422, stage="history_compression")

    latest_artifact_type = latest_artifact.get("artifact_type") if isinstance(latest_artifact, dict) else None
    available_artifact_types = [latest_artifact_type.strip()] if isinstance(latest_artifact_type, str) and latest_artifact_type.strip() else []

    routing_decision = plan_request_with_llm(
        latest_user_message,
        dataset_columns=dataset_columns,
        prior_analysis_active=prior_analysis_active,
        dataset_summary=dataset_summary,
        schema_profile=schema_profile,
        latest_artifact=latest_artifact,
        available_artifact_types=available_artifact_types,
        recommended_prompts=recommended_prompts,
    )

    # Safe fallback if LLM is unavailable or JSON parsing fails
    if routing_decision is None:
        routing_decision = RoutingDecision(
            primary_mode="analysis" if dataset_id else "direct_answer",
            needs_dataset=bool(dataset_id),
            needs_tool_execution=bool(dataset_id),
            route_source="fallback",
        )

    # Check if dataset is required but missing
    if routing_decision.needs_dataset and not dataset_id:
        raise AppError(
            "dataset_required",
            "No dataset selected. Please upload a CSV file first.",
            400,
            stage="validation",
        )

    final_branch = resolve_final_branch(
        dataset_id=dataset_id,
        routing_decision=routing_decision,
    )

    return ChatRequestRequirements(
        dataset_id=dataset_id,
        messages=messages,
        compressed_messages=compressed.messages_for_model,
        latest_user_message=latest_user_message,
        prior_analysis_active=prior_analysis_active,
        routing_decision=routing_decision,
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
            "primary_mode": requirements.routing_decision.primary_mode,
            "route_source": requirements.routing_decision.route_source,
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
