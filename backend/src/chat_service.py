from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Request
from langgraph.errors import GraphRecursionError

from src.agent import AgentContext, generate_general_chat_reply, get_dataset_required_decision
from src.conversation_context import compress_conversation_messages
from src.data_manager import get_dataset
from src.errors import AppError
from src.request_context import get_degradation_mode, get_request_id, set_degradation_mode, set_failure_stage
from src.request_parsing import (
    extract_dataset_id_from_payload,
    extract_latest_user_message,
    extract_messages,
    has_prior_analysis_context,
)
from src.result_types import artifact_registry
from src.routing_rules import RoutingContext, interpret_request
from src.settings import SETTINGS
from src.sse import backend_image_url, build_streaming_response, extract_text_from_chunk, format_sse, now_iso
from src.tools import bind_current_dataset_id, consume_current_image_event

logger = logging.getLogger(__name__)

ML_RESULT_ARTIFACT_TYPES = {"model_result", "metrics_result", "feature_importance_result"}
ML_DIRECT_TOOL_NAMES = {"ml_execute"}

FOLLOW_UP_MARKERS = (
    "解释",
    "刚才",
    "之前",
    "上一个",
    "上个",
    "这个结果",
    "这结果",
    "上面的结果",
    "继续",
    "follow up",
    "follow-up",
)

CHART_MARKERS = (
    "画图",
    "绘图",
    "生成图",
    "生成一个图",
    "图表",
    "柱状图",
    "折线图",
    "散点图",
    "直方图",
    "热力图",
    "可视化",
    "chart",
    "plot",
    "visualize",
)

DATASET_OVERVIEW_MARKERS = (
    "讲解数据集",
    "介绍数据集",
    "解释数据集",
    "数据集概览",
    "数据集说明",
    "看看数据集",
    "了解数据集",
    "describe dataset",
    "explain dataset",
    "summarize dataset",
    "dataset overview",
)

ML_TRAINING_TERMS = (
    "train a model",
    "train a logistic regression model",
    "train a linear regression model",
    "train model",
    "build a model",
    "build model",
    "fit model",
    "baseline model",
    "classifier",
    "classification model",
    "logistic regression",
    "linear regression",
    "predict",
    "prediction",
    "forecast",
    "训练模型",
    "训练一个模型",
    "训练一个 baseline",
    "分类器",
    "分类模型",
    "逻辑回归",
    "线性回归",
    "预测",
    "预测一下",
)

ML_METRICS_TERMS = (
    "model metrics",
    "metrics",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "roc auc",
    "模型指标",
    "准确率",
    "精确率",
    "召回率",
)

ML_IMPORTANCE_TERMS = (
    "feature importance",
    "coefficients",
    "coefficient",
    "特征重要性",
    "重要特征",
    "系数",
)


@dataclass(slots=True)
class ChatRequestRequirements:
    dataset_id: str | None
    messages: list[dict[str, object]]
    compressed_messages: list[dict[str, str]]
    simple_mode_messages: list[dict[str, str]]
    latest_user_message: str
    prior_analysis_active: bool
    interpretation_intent: str
    chart_requested: bool
    explicit_ml_request: bool
    required_ml_artifacts: set[str]
    follow_up_message: dict[str, object] | None
    is_dataset_overview: bool
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
    repeated_tool_name: str | None = None
    repeated_tool_steps: int = 0
    no_progress_steps: int = 0
    progress_since_last_tool: bool = False

    def register_tool_start(self, tool_name: str | None) -> None:
        self.total_steps += 1
        if tool_name and tool_name == self.repeated_tool_name:
            self.repeated_tool_steps += 1
        else:
            self.repeated_tool_name = tool_name
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


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _contains_any_term(value: str, terms: tuple[str, ...]) -> bool:
    return any(term in value for term in terms)


def _looks_like_follow_up_request(message: str) -> bool:
    return _contains_any_term(_normalize_text(message), tuple(marker.lower() for marker in FOLLOW_UP_MARKERS))


def _looks_like_chart_request(message: str) -> bool:
    return _contains_any_term(_normalize_text(message), tuple(marker.lower() for marker in CHART_MARKERS))


def _looks_like_explicit_ml_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if not normalized:
        return False
    if _looks_like_follow_up_request(normalized) and any(
        token in normalized for token in ("model", "模型", "指标", "metrics", "feature importance", "重要特征")
    ):
        return True
    return _contains_any_term(normalized, ML_TRAINING_TERMS + ML_METRICS_TERMS + ML_IMPORTANCE_TERMS)


def _collect_required_ml_artifacts(message: str) -> set[str]:
    normalized = _normalize_text(message)
    required: set[str] = set()
    if _contains_any_term(normalized, ML_TRAINING_TERMS):
        required.add("model_result")
    if _contains_any_term(normalized, ML_METRICS_TERMS):
        required.add("metrics_result")
    if _contains_any_term(normalized, ML_IMPORTANCE_TERMS):
        required.add("feature_importance_result")
    return required


def _build_follow_up_context_message(dataset_id: str, latest_user_message: str) -> dict[str, object] | None:
    if not _looks_like_follow_up_request(latest_user_message):
        return None

    latest_artifact = artifact_registry.get_latest(dataset_id)
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
        "requires_ml",
        "requires_chart",
        "requires_python_analysis",
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


def _looks_like_dataset_overview_request(message: str) -> bool:
    return _contains_any_term(_normalize_text(message), DATASET_OVERVIEW_MARKERS)


def _format_column_list(columns: list[str], *, limit: int = 8) -> str:
    if not columns:
        return "暂无"
    visible_columns = columns[:limit]
    suffix = f" 等 {len(columns)} 列" if len(columns) > limit else ""
    return "、".join(visible_columns) + suffix


def _build_recommended_dataset_questions(column_names: set[str]) -> list[str]:
    if {"order_date", "total_amount", "product_category", "region", "channel"}.issubset(column_names):
        return [
            "每月销售额趋势是什么？请画一张折线图。",
            "哪个商品品类收入最高？请按区域对比。",
            "比较线上和线下渠道的 total_amount，做一个 t 检验。",
        ]
    if {"study_hours", "attendance_rate", "final_score", "gender"}.issubset(column_names):
        return [
            "study_hours 和 final_score 的相关性是多少？",
            "用 study_hours 和 attendance_rate 预测 final_score，跑一个线性回归。",
            "男女学生成绩是否有显著差异？请做 t 检验。",
        ]
    if {"conversion_flag", "ab_group", "channel_source", "session_count"}.issubset(column_names):
        return [
            "比较 A/B 组的 conversion_flag 转化率，并做卡方检验。",
            "哪个 channel_source 的转化率最高？",
            "按 session_count 给用户分层，并可视化分布。",
        ]
    return [
        "请先做一份描述性统计，并指出值得关注的字段。",
        "哪些数值字段之间可能存在相关性？",
        "按一个关键分类字段分组，比较主要指标差异。",
    ]


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
    return AppError(
        "internal_error",
        "服务器内部错误，请稍后重试。",
        500,
        retryable=False,
        stage="unknown",
    )


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


def build_dataset_overview_reply(dataset: object) -> str:
    columns = getattr(dataset, "columns", [])
    column_names = [
        str(column.get("name"))
        for column in columns
        if isinstance(column, dict) and column.get("name")
    ]
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
    recommendation_lines = _build_recommended_dataset_questions(set(column_names))

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
    lines.extend(f"- {question}" for question in recommendation_lines)
    return "\n".join(lines)


def analyze_chat_request(payload: dict[str, object]) -> ChatRequestRequirements:
    dataset_id = extract_dataset_id_from_payload(payload)
    messages = extract_messages(payload)
    if not messages:
        raise AppError("validation_error", "请求参数不合法，请先输入问题。", 422, stage="validation")

    latest_user_message = extract_latest_user_message(messages)
    prior_analysis_active = has_prior_analysis_context(messages)

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

    compressed = compress_conversation_messages(messages, dataset_id=dataset_id)
    if not compressed.messages_for_model:
        raise AppError("history_compression_error", "会话历史压缩失败，请重新发起请求。", 422, stage="history_compression")

    interpretation = interpret_request(
        RoutingContext(
            message=latest_user_message,
            prior_analysis_active=prior_analysis_active,
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
        interpretation_intent=interpretation.intent_type,
        chart_requested=_looks_like_chart_request(latest_user_message),
        explicit_ml_request=_looks_like_explicit_ml_request(latest_user_message),
        required_ml_artifacts=_collect_required_ml_artifacts(latest_user_message),
        follow_up_message=_build_follow_up_context_message(dataset_id, latest_user_message) if dataset_id else None,
        is_dataset_overview=bool(dataset_id and _looks_like_dataset_overview_request(latest_user_message)),
        history_message_count=compressed.history_message_count,
        compressed_history_count=compressed.compressed_history_count,
        artifact_refs_count=compressed.artifact_refs_count,
        conversation_summary=compressed.conversation_summary,
    )


async def _stream_graph_attempt(
    request: Request,
    *,
    graph: Any,
    requirements: ChatRequestRequirements,
    dataset_id: str,
    messages_for_graph: list[dict[str, object]],
    recursion_limit: int,
) -> tuple[list[str], StreamOutcome]:
    set_failure_stage("model_stream")
    runtime_context = AgentContext(dataset_id=dataset_id)
    prior_model_artifact_ids = {
        artifact_type: (
            artifact_registry.get_latest(dataset_id, artifact_type=artifact_type) or {}
        ).get("artifact_id")
        for artifact_type in ML_RESULT_ARTIFACT_TYPES
    }
    outcome = StreamOutcome(buffered_text_chunks=[])
    emitted_events: list[str] = []
    loop_guard = LoopGuardState()
    async for event in graph.astream_events(
        {"messages": messages_for_graph},
        config={
            "configurable": {"dataset_id": dataset_id},
            "recursion_limit": recursion_limit,
        },
        context=runtime_context,
        version="v2",
    ):
        event_name = str(event.get("event") or "")
        data = event.get("data") or {}
        name = event.get("name")
        if event_name == "on_tool_start":
            loop_guard.register_tool_start(str(name) if isinstance(name, str) else None)
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
                    current_artifact = artifact_registry.get_latest(dataset_id, artifact_type=artifact_type)
                    prior_artifact_id = prior_model_artifact_ids.get(artifact_type)
                    current_artifact_id = current_artifact.get("artifact_id") if current_artifact else None
                    if current_artifact_id and current_artifact_id != prior_artifact_id:
                        outcome.produced_ml_artifact_types.add(str(artifact_type))
                        loop_guard.register_artifact_progress()
            else:
                emitted_events.append(
                    format_sse(
                        "message_chunk",
                        _build_event_payload({"content": text}, dataset_id=dataset_id),
                    )
                )
            continue

        if event_name == "on_tool_start":
            emitted_events.append(
                format_sse(
                    "tool_start",
                    _build_event_payload({"tool_name": name}, dataset_id=dataset_id),
                )
            )
            continue

        if event_name == "on_tool_end":
            if outcome.tool_error_payload is not None:
                _raise_tool_error(outcome.tool_error_payload)
            if isinstance(name, str) and name in ML_DIRECT_TOOL_NAMES:
                for artifact_type in ML_RESULT_ARTIFACT_TYPES:
                    current_artifact = artifact_registry.get_latest(dataset_id, artifact_type=artifact_type)
                    prior_artifact_id = prior_model_artifact_ids.get(artifact_type)
                    if current_artifact is not None and current_artifact.get("artifact_id") != prior_artifact_id:
                        outcome.produced_ml_artifact_types.add(str(artifact_type))
                        loop_guard.register_artifact_progress()

            emitted_events.append(
                format_sse(
                    "tool_end",
                    _build_event_payload({"tool_name": name}, dataset_id=dataset_id),
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
                            _build_event_payload(
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
                _build_event_payload({"content": text}, dataset_id=dataset_id),
            )
        )
    return emitted_events, outcome


async def create_chat_stream_response(
    request: Request,
    payload: dict[str, object],
    *,
    graph: Any,
) -> Any:
    requirements = analyze_chat_request(payload)
    dataset_id = requirements.dataset_id

    logger.info(
        "chat_stream payload received",
        extra={
            "request_id": get_request_id(),
            "dataset_id": dataset_id,
            "message_preview": requirements.latest_user_message[:80],
            "intent_type": requirements.interpretation_intent,
            "history_message_count": requirements.history_message_count,
            "compressed_history_count": requirements.compressed_history_count,
            "artifact_refs_count": requirements.artifact_refs_count,
            "degradation_mode": get_degradation_mode(),
        },
    )

    if not dataset_id:
        async def general_chat_event_generator():
            attempts = [
                ("none", requirements.compressed_messages),
                ("retry_stream_once", requirements.compressed_messages),
                ("non_stream_simple_mode", requirements.simple_mode_messages),
            ]
            for index, (mode, messages_for_reply) in enumerate(attempts):
                try:
                    set_degradation_mode(mode)
                    set_failure_stage("model_stream")
                    reply = await generate_general_chat_reply(messages_for_reply)
                    if reply.strip():
                        yield format_sse(
                            "message_chunk",
                            _build_event_payload({"content": reply}),
                        )
                    yield _build_done_sse()
                    return
                except Exception as exc:
                    mapped_error = _classify_stream_exception(exc)
                    logger.exception(
                        "general chat stream failed",
                        extra={
                            "request_id": get_request_id(),
                            "attempt": index,
                            "degradation_mode": mode,
                            "error_code": mapped_error.code,
                            "failure_stage": mapped_error.stage,
                        },
                    )
                    if mapped_error.retryable and index < len(attempts) - 1:
                        continue
                    yield _build_error_sse(mapped_error)
                    yield _build_done_sse()
                    return

        return build_streaming_response(general_chat_event_generator())

    dataset = get_dataset(dataset_id)
    if requirements.is_dataset_overview:
        async def dataset_overview_event_generator():
            with bind_current_dataset_id(dataset_id):
                yield format_sse(
                    "message_chunk",
                    _build_event_payload(
                        {"content": build_dataset_overview_reply(dataset)},
                        dataset_id=dataset_id,
                    ),
                )
                yield _build_done_sse(dataset_id=dataset_id)

        return build_streaming_response(dataset_overview_event_generator())

    async def event_generator():
        with bind_current_dataset_id(dataset_id):
            attempts = [
                ("none", requirements.messages_for_graph, SETTINGS.agent_default_recursion_limit),
                ("retry_stream_once", requirements.messages_for_graph, SETTINGS.agent_default_recursion_limit),
                (
                    "non_stream_simple_mode",
                    requirements.simple_messages_for_graph,
                    SETTINGS.agent_simple_mode_recursion_limit,
                ),
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
                        )
                        for event in events:
                            yield event
                        yield _build_done_sse(dataset_id=dataset_id)
                        return
                    except Exception as exc:
                        mapped_error = _classify_stream_exception(exc)
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
                                "intent_type": requirements.interpretation_intent,
                                "failure_stage": mapped_error.stage,
                                "error_code": mapped_error.code,
                                "attempt": index,
                            },
                        )
                        if mapped_error.retryable and index < len(attempts) - 1:
                            continue
                        if mapped_error.code == "agent_recursion_limit" and index < len(attempts) - 1:
                            continue
                        yield _build_error_sse(mapped_error, dataset_id=dataset_id)
                        yield _build_done_sse(dataset_id=dataset_id)
                        return
            except AppError as exc:
                set_failure_stage(exc.stage or "unknown")
                yield _build_error_sse(exc, dataset_id=dataset_id)
                yield _build_done_sse(dataset_id=dataset_id)
            except Exception as exc:
                mapped_error = _classify_stream_exception(exc)
                set_failure_stage(mapped_error.stage)
                logger.exception(
                    "chat stream failed",
                    extra={
                        "request_id": get_request_id(),
                        "dataset_id": dataset_id,
                        "failure_stage": mapped_error.stage,
                        "error_code": mapped_error.code,
                    },
                )
                yield _build_error_sse(mapped_error, dataset_id=dataset_id)
                yield _build_done_sse(dataset_id=dataset_id)

    return build_streaming_response(event_generator())


def _validate_stream_outcome(requirements: ChatRequestRequirements, outcome: StreamOutcome) -> None:
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
