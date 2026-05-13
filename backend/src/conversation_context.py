from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.result_types import artifact_registry
from src.settings import SETTINGS


@dataclass(slots=True)
class CompressedConversation:
    messages_for_model: list[dict[str, str]]
    history_message_count: int
    compressed_history_count: int
    artifact_refs_count: int
    conversation_summary: dict[str, Any] | None
    summary_message: dict[str, str] | None


def _normalize_line(value: str, *, max_length: int) -> str:
    return " ".join(value.strip().split())[:max_length]


def _looks_like_user_goal(message: str) -> bool:
    normalized = message.lower()
    return any(
        token in normalized
        for token in (
            "请",
            "分析",
            "compare",
            "group",
            "plot",
            "chart",
            "预测",
            "训练",
            "explain",
            "为什么",
        )
    )


def _build_artifact_reference(dataset_id: str | None) -> dict[str, Any] | None:
    if not dataset_id:
        return None
    latest_artifact = artifact_registry.get_latest(dataset_id)
    if not isinstance(latest_artifact, dict):
        return None
    return {
        "artifact_id": latest_artifact.get("artifact_id"),
        "artifact_type": latest_artifact.get("artifact_type"),
        "warnings": latest_artifact.get("warnings", []),
        "target": latest_artifact.get("target"),
        "model_type": latest_artifact.get("model_type"),
    }


def _append_unique(items: list[str], value: str, *, limit: int) -> None:
    normalized = value.strip()
    if not normalized or normalized in items:
        return
    items.append(normalized[: SETTINGS.chat_history_summary_max_chars])
    if len(items) > limit:
        del items[0]


def _looks_like_constraint(message: str) -> bool:
    normalized = message.lower()
    return any(
        token in normalized
        for token in (
            "只看",
            "仅看",
            "filter",
            "where",
            "按",
            "限定",
            "范围",
            "时间",
            "state",
            "region",
            "channel",
            "california",
            "texas",
            "new york",
        )
    )


def _looks_like_metric_or_target(message: str) -> bool:
    normalized = message.lower()
    return any(
        token in normalized
        for token in (
            "sales",
            "orders",
            "churn",
            "target",
            "指标",
            "目标",
            "accuracy",
            "auc",
            "f1",
            "回归",
            "逻辑回归",
            "线性回归",
            "plot",
            "chart",
            "图",
        )
    )


def _build_simple_mode_messages(compressed_messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not compressed_messages:
        return []
    summary_message = compressed_messages[0] if "会话摘要" in compressed_messages[0].get("content", "") else None
    latest_user_message: dict[str, str] | None = None
    latest_assistant_message: dict[str, str] | None = None
    for message in reversed(compressed_messages):
        role = str(message.get("type", "user"))
        if latest_user_message is None and role in {"human", "user"}:
            latest_user_message = message
            continue
        if latest_assistant_message is None and role in {"ai", "assistant"} and message is not summary_message:
            latest_assistant_message = message
        if latest_user_message is not None and latest_assistant_message is not None:
            break

    simple_messages: list[dict[str, str]] = []
    if summary_message is not None:
        simple_messages.append(summary_message)
        if latest_assistant_message is not None:
            simple_messages.append(latest_assistant_message)
        if latest_user_message is not None:
            simple_messages.append(latest_user_message)
        return simple_messages

    return list(compressed_messages[-2:])


def compress_conversation_messages(
    messages: list[dict[str, object]],
    *,
    dataset_id: str | None,
    keep_recent: int | None = None,
) -> CompressedConversation:
    keep_recent_count = keep_recent or SETTINGS.chat_history_keep_recent_messages
    text_messages = [
        message
        for message in messages
        if isinstance(message.get("content"), str) and str(message.get("content")).strip()
    ]
    history_message_count = len(text_messages)
    if history_message_count <= keep_recent_count:
        return CompressedConversation(
            messages_for_model=[
                {"type": str(message.get("type", "user")), "content": str(message.get("content", ""))}
                for message in text_messages
            ],
            history_message_count=history_message_count,
            compressed_history_count=0,
            artifact_refs_count=1 if _build_artifact_reference(dataset_id) else 0,
            conversation_summary=None,
            summary_message=None,
        )

    preserved_messages = text_messages[-keep_recent_count:]
    older_messages = text_messages[:-keep_recent_count]
    global_constraints: list[str] = []
    current_goals: list[str] = []
    recent_assistant_findings: list[str] = []
    unresolved_followups: list[str] = []
    for message in older_messages:
        role = str(message.get("type", "user"))
        content = _normalize_line(str(message.get("content", "")), max_length=SETTINGS.chat_history_summary_max_chars)
        if not content:
            continue
        if role in {"human", "user"}:
            if _looks_like_constraint(content):
                _append_unique(global_constraints, content, limit=6)
            if _looks_like_metric_or_target(content) or _looks_like_user_goal(content):
                _append_unique(current_goals, content, limit=6)
        elif role in {"ai", "assistant"} and _looks_like_metric_or_target(content):
            _append_unique(recent_assistant_findings, content, limit=4)

    for message in older_messages[-12:]:
        role = str(message.get("type", "user"))
        content = _normalize_line(str(message.get("content", "")), max_length=SETTINGS.chat_history_summary_max_chars)
        if not content:
            continue
        if role in {"human", "user"}:
            if _looks_like_user_goal(content):
                _append_unique(current_goals, content, limit=6)
            else:
                _append_unique(unresolved_followups, content, limit=3)
        elif role in {"ai", "assistant"}:
            _append_unique(recent_assistant_findings, content, limit=4)

    artifact_reference = _build_artifact_reference(dataset_id)
    summary_payload: dict[str, Any] = {
        "summary_type": "conversation_summary",
        "dataset_id": dataset_id,
        "older_message_count": len(older_messages),
        "global_constraints": global_constraints[-4:],
        "current_goals": current_goals[-4:],
        "recent_assistant_findings": recent_assistant_findings[-4:],
        "open_followups": unresolved_followups[-3:],
    }
    if artifact_reference is not None:
        summary_payload["latest_artifact_reference"] = artifact_reference

    summary_message = {
        "type": "assistant",
        "content": "会话摘要（供后续推理复用，不要逐字复述）：\n" + json.dumps(summary_payload, ensure_ascii=False),
    }
    messages_for_model = [summary_message]
    messages_for_model.extend(
        {"type": str(message.get("type", "user")), "content": str(message.get("content", ""))}
        for message in preserved_messages
    )
    return CompressedConversation(
        messages_for_model=messages_for_model,
        history_message_count=history_message_count,
        compressed_history_count=len(older_messages),
        artifact_refs_count=1 if artifact_reference else 0,
        conversation_summary=summary_payload,
        summary_message=summary_message,
    )
