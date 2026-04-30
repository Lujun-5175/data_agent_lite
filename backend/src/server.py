from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent import graph
from src.agent import AgentContext
from src.agent import generate_general_chat_reply
from src.agent import get_dataset_required_decision
from src.data_manager import (
    DatasetLoadError,
    DatasetNotFoundError,
    calculate_correlation,
    cleanup_dataset_artifacts,
    cleanup_expired_artifacts,
    get_data_preview,
    get_dataset,
    load_csv_file,
)
from src.errors import AppError
from src.result_types import artifact_registry
from src.routing_rules import RoutingContext, interpret_request
from src.settings import SETTINGS
from src.tools import consume_current_image_event, set_current_dataset_id

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = (BACKEND_ROOT / "static").resolve()
IMAGES_DIR = (STATIC_DIR / "images").resolve()
TEMP_DATA_DIR = (BACKEND_ROOT / "temp_data").resolve()
MAX_UPLOAD_SIZE_BYTES = SETTINGS.max_upload_size_bytes
UPLOAD_CHUNK_SIZE = SETTINGS.upload_chunk_size
ARTIFACT_CLEANUP_INTERVAL_SECONDS = SETTINGS.artifact_cleanup_interval_seconds
PRODUCTION_ENV_MARKERS = (
    "RAILWAY_ENVIRONMENT",
    "RAILWAY_ENVIRONMENT_NAME",
    "RAILWAY_PROJECT_ID",
    "RENDER",
    "FLY_APP_NAME",
)


def _default_app_env() -> str:
    if any(os.getenv(name) for name in PRODUCTION_ENV_MARKERS):
        return "production"
    return "development"


APP_ENV = os.getenv("APP_ENV", os.getenv("ENV", os.getenv("FASTAPI_ENV", _default_app_env()))).strip().lower()
IS_DEVELOPMENT = APP_ENV in {"dev", "development", "local", "test"}
DEV_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

STATIC_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_cors_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    if IS_DEVELOPMENT:
        # Keep local development usable out of the box with explicit local origins.
        return DEV_CORS_ORIGINS
    raise RuntimeError("CORS_ALLOW_ORIGINS must be configured when APP_ENV is production.")


class CorrelationRequest(BaseModel):
    dataset_id: str = Field(description="数据集 ID")
    col1: str = Field(description="第一列名称")
    col2: str = Field(description="第二列名称")


class ErrorPayload(BaseModel):
    code: str
    message: str


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": ErrorPayload(code=code, message=message).model_dump()},
        headers={"X-Error-Code": code},
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_sse(event_type: str, payload: dict[str, object]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _extract_dataset_id_from_payload(payload: dict[str, object]) -> str | None:
    config = payload.get("config")
    if isinstance(config, dict):
        configurable = config.get("configurable")
        if isinstance(configurable, dict):
            dataset_id = configurable.get("dataset_id")
            if isinstance(dataset_id, str) and dataset_id.strip():
                return dataset_id.strip()
    dataset_id = payload.get("dataset_id")
    if isinstance(dataset_id, str) and dataset_id.strip():
        return dataset_id.strip()
    return None


def _extract_messages(payload: dict[str, object]) -> list[dict[str, object]]:
    input_payload = payload.get("input")
    if isinstance(input_payload, dict):
        messages = input_payload.get("messages")
        if isinstance(messages, list):
            return [message for message in messages if isinstance(message, dict)]
    messages = payload.get("messages")
    if isinstance(messages, list):
        return [message for message in messages if isinstance(message, dict)]
    return []


def _extract_latest_user_message(messages: list[dict[str, object]]) -> str:
    for message in reversed(messages):
        role = message.get("type")
        if role not in {"human", "user"}:
            continue
        content = message.get("content")
        if isinstance(content, str):
            stripped = content.strip()
            if stripped:
                return stripped
    return ""


def _has_prior_analysis_context(messages: list[dict[str, object]]) -> bool:
    for message in messages[:-1]:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        normalized = content.lower()
        if any(token in normalized for token in ("dataset", "数据", "统计", "相关性", "group", "检验", "analysis")):
            return True
    return False


def _extract_text_from_chunk(chunk: object) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        content = chunk.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(_extract_text_from_chunk(item) for item in content)
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            parts.append(_extract_text_from_chunk(item))
        return "".join(parts)
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    return ""


def _backend_image_url(request: Request, filename: str) -> str:
    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/static/images/{filename}"


ML_RESULT_ARTIFACT_TYPES = {"model_result", "metrics_result", "feature_importance_result"}

ML_DIRECT_TOOL_NAMES = {
    "ml_execute",
}

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


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _looks_like_follow_up_request(message: str) -> bool:
    normalized = _normalize_text(message)
    return any(marker.lower() in normalized for marker in FOLLOW_UP_MARKERS)


def _looks_like_chart_request(message: str) -> bool:
    normalized = _normalize_text(message)
    return any(marker.lower() in normalized for marker in CHART_MARKERS)


_ML_EXPLICIT_TERMS = (
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
    "evaluate model",
    "model metrics",
    "metrics",
    "feature importance",
    "coefficients",
    "coefficient",
    "训练模型",
    "训练一个模型",
    "训练一个 baseline",
    "baseline model",
    "分类器",
    "分类模型",
    "逻辑回归",
    "线性回归",
    "预测",
    "预测一下",
    "模型指标",
    "特征重要性",
    "重要特征",
    "准确率",
    "精确率",
    "召回率",
    "f1",
    "roc auc",
)


def _looks_like_explicit_ml_request(message: str, *, prior_analysis_active: bool = False) -> bool:
    normalized = _normalize_text(message)
    if not normalized:
        return False
    if _looks_like_follow_up_request(normalized) and any(
        token in normalized for token in ("model", "模型", "指标", "metrics", "feature importance", "重要特征")
    ):
        return True
    return any(term in normalized for term in _ML_EXPLICIT_TERMS)


def _looks_like_explicit_chart_request(message: str) -> bool:
    return _looks_like_chart_request(message)


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
        "content": (
            "最近一次结构化结果，供解释或跟进使用：\n"
            f"{json.dumps(summary, ensure_ascii=False)}"
        ),
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
    leading_whitespace_len = len(text) - len(text.lstrip())
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


def _looks_like_dataset_overview_request(message: str) -> bool:
    normalized = _normalize_text(message)
    return any(marker in normalized for marker in DATASET_OVERVIEW_MARKERS)


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


def _build_dataset_overview_reply(dataset: object) -> str:
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


app = FastAPI(
    title="Data Agent Backend",
    version="1.1",
    description="Data Agent Backend with safer dataset handling",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


async def _artifact_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(ARTIFACT_CLEANUP_INTERVAL_SECONDS)
        try:
            summary = cleanup_expired_artifacts(temp_dir=TEMP_DATA_DIR, images_dir=IMAGES_DIR)
            logger.info("Expired artifacts cleaned", extra=summary)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Artifact cleanup loop failed")


@app.on_event("startup")
async def startup_artifact_cleanup() -> None:
    summary = cleanup_expired_artifacts(temp_dir=TEMP_DATA_DIR, images_dir=IMAGES_DIR)
    logger.info("Startup artifact cleanup completed", extra=summary)
    app.state.artifact_cleanup_task = asyncio.create_task(_artifact_cleanup_loop())


@app.on_event("shutdown")
async def shutdown_artifact_cleanup() -> None:
    task = getattr(app.state, "artifact_cleanup_task", None)
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@app.exception_handler(DatasetNotFoundError)
async def handle_dataset_not_found(request: Request, exc: DatasetNotFoundError) -> JSONResponse:
    return error_response(exc.status_code, exc.code, exc.message)


@app.exception_handler(DatasetLoadError)
async def handle_dataset_load_error(request: Request, exc: DatasetLoadError) -> JSONResponse:
    return error_response(exc.status_code, exc.code, exc.message)


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    return error_response(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(422, "validation_error", "请求参数不合法，请检查后重试。")


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server exception")
    return error_response(500, "internal_error", "服务器内部错误，请稍后重试。")


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "message": "Data Agent Backend is running"}


@app.get("/data-preview")
async def data_preview(dataset_id: str) -> dict[str, object]:
    dataset = get_dataset(dataset_id)
    return {
        "status": "success",
        "dataset_id": dataset.dataset_id,
        "preview": get_data_preview(dataset.dataset_id),
        "original_filename": dataset.original_filename,
        "original_row_count": dataset.original_row_count,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "preview_count": dataset.preview_count,
        "columns": dataset.columns,
        "filename": dataset.original_filename,
        "analysis_basis": dataset.analysis_basis,
        "preprocessed": dataset.preprocessed,
        "preprocessing_log": dataset.preprocessing_log,
        "schema_profile": dataset.schema_profile_artifact,
        "analysis_preprocess": dataset.analysis_preprocess_artifact,
        "model_prep_plan": dataset.model_prep_plan_artifact,
    }


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)) -> JSONResponse:
    original_filename = file.filename or "uploaded.csv"
    if not original_filename.lower().endswith(".csv"):
        raise AppError("invalid_file_type", "只支持 CSV 文件上传。", 400)

    safe_filename = f"{uuid4().hex}.csv"
    stored_path = (TEMP_DATA_DIR / safe_filename).resolve()
    temp_dir_resolved = TEMP_DATA_DIR.resolve()

    if temp_dir_resolved != stored_path.parent:
        raise AppError("invalid_file_type", "上传路径不安全，已拒绝请求。", 400)

    bytes_written = 0

    try:
        with stored_path.open("wb") as target:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_SIZE_BYTES:
                    raise AppError("file_too_large", "上传文件超过 50MB 限制。", 413)
                target.write(chunk)

        dataset = load_csv_file(stored_path, original_filename)
        response = {
            "status": "success",
            "message": f"成功加载文件【{dataset.original_filename}】！包含 {dataset.row_count} 行，{len(dataset.columns)} 列。",
            "dataset_id": dataset.dataset_id,
            "original_filename": dataset.original_filename,
            "preview": dataset.preview,
            "analysis_basis": dataset.analysis_basis,
            "preprocessed": dataset.preprocessed,
            "preprocessing_log": dataset.preprocessing_log,
            "original_row_count": dataset.original_row_count,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
            "preview_count": dataset.preview_count,
            "columns": dataset.columns,
            "filename": dataset.original_filename,
            "schema_profile": dataset.schema_profile_artifact,
            "analysis_preprocess": dataset.analysis_preprocess_artifact,
            "model_prep_plan": dataset.model_prep_plan_artifact,
        }
        return JSONResponse(content=response)
    except AppError:
        if stored_path.exists():
            stored_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        logger.exception("Unexpected upload failure")
        if stored_path.exists():
            stored_path.unlink(missing_ok=True)
        raise AppError("internal_error", "文件上传失败，请稍后重试。", 500) from exc
    finally:
        await file.close()


@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str) -> dict[str, object]:
    cleanup_dataset_artifacts(dataset_id)

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "message": "数据集已删除。",
    }


@app.post("/calculate-correlation")
async def get_correlation(request: CorrelationRequest) -> dict[str, object]:
    result = calculate_correlation(request.dataset_id, request.col1, request.col2)
    return result


@app.post("/chat/stream", response_model=None)
async def chat_stream(request: Request) -> StreamingResponse | JSONResponse:
    try:
        payload = await request.json()
    except Exception as exc:
        raise AppError("validation_error", "请求参数不合法，请检查后重试。", 422) from exc

    if not isinstance(payload, dict):
        raise AppError("validation_error", "请求参数不合法，请检查后重试。", 422)

    dataset_id = _extract_dataset_id_from_payload(payload)
    messages = _extract_messages(payload)
    if not messages:
        raise AppError("validation_error", "请求参数不合法，请先输入问题。", 422)

    latest_user_message = _extract_latest_user_message(messages)
    logger.info(
        "chat_stream payload received",
        extra={
            "dataset_id": dataset_id,
            "message_preview": latest_user_message[:80],
        },
    )
    dataset_required_decision = get_dataset_required_decision(
        latest_user_message,
        dataset_columns=[],
        prior_analysis_active=_has_prior_analysis_context(messages),
    )
    logger.debug("dataset-required decision: %s", dataset_required_decision.to_dict())
    if not dataset_id and dataset_required_decision.matched:
        raise AppError("dataset_required", "当前未选择数据集，请先上传 CSV 文件后再进行数据分析。", 400)

    if not dataset_id:
        async def general_chat_event_generator():
            try:
                reply = await generate_general_chat_reply(messages)
                if reply.strip():
                    yield _format_sse(
                        "message_chunk",
                        {
                            "content": reply,
                            "timestamp": _now_iso(),
                        },
                    )
                yield _format_sse("done", {"timestamp": _now_iso()})
            except AppError as exc:
                yield _format_sse("error", {"code": exc.code, "message": exc.message})
                yield _format_sse("done", {"timestamp": _now_iso()})
            except Exception:
                logger.exception("general chat stream failed")
                yield _format_sse("error", {"code": "internal_error", "message": "服务器内部错误，请稍后重试。"})
                yield _format_sse("done", {"timestamp": _now_iso()})

        return StreamingResponse(
            general_chat_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    set_current_dataset_id(dataset_id)

    dataset = get_dataset(dataset_id)
    if _looks_like_dataset_overview_request(latest_user_message):
        async def dataset_overview_event_generator():
            try:
                yield _format_sse(
                    "message_chunk",
                    {
                        "content": _build_dataset_overview_reply(dataset),
                        "dataset_id": dataset_id,
                        "timestamp": _now_iso(),
                    },
                )
                yield _format_sse("done", {"dataset_id": dataset_id, "timestamp": _now_iso()})
            finally:
                set_current_dataset_id(None)

        return StreamingResponse(
            dataset_overview_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    prior_analysis_active = _has_prior_analysis_context(messages)
    interpretation = interpret_request(
        RoutingContext(
            message=latest_user_message,
            prior_analysis_active=prior_analysis_active,
        )
    )
    chart_requested = _looks_like_explicit_chart_request(latest_user_message)
    explicit_ml_request = _looks_like_explicit_ml_request(latest_user_message, prior_analysis_active=prior_analysis_active)
    normalized_latest_user_message = _normalize_text(latest_user_message)
    ml_needs_training = any(
        term in _normalize_text(latest_user_message)
        for term in (
            "train a model",
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
            "baseline model",
            "分类器",
            "分类模型",
            "逻辑回归",
            "线性回归",
            "预测",
            "预测一下",
        )
    )
    ml_needs_metrics = any(
        term in normalized_latest_user_message
        for term in ("model metrics", "metrics", "accuracy", "precision", "recall", "f1", "auc", "roc auc", "模型指标", "准确率", "精确率", "召回率", "f1", "roc auc")
    )
    ml_needs_feature_importance = any(
        term in normalized_latest_user_message
        for term in ("feature importance", "coefficients", "coefficient", "特征重要性", "重要特征", "系数")
    )
    follow_up_message = _build_follow_up_context_message(dataset_id, latest_user_message)
    messages_for_graph = list(messages)
    if follow_up_message is not None:
        messages_for_graph.append(follow_up_message)

    async def event_generator():
        try:
            runtime_context = AgentContext(dataset_id=dataset_id)
            prior_model_artifact_ids: dict[str, str | None] = {}
            for artifact_type in ML_RESULT_ARTIFACT_TYPES:
                prior_artifact = artifact_registry.get_latest(dataset_id, artifact_type=artifact_type)
                prior_model_artifact_ids[artifact_type] = prior_artifact.get("artifact_id") if prior_artifact else None
            buffered_text_chunks: list[str] = []
            saw_ml_result = False
            saw_chart_image = False
            saw_ml_tool_call = False
            produced_ml_artifact_types: set[str] = set()
            logger.debug(
                "starting chat stream with runtime context",
                extra={
                    "dataset_id": dataset_id,
                    "has_runtime_context": True,
                    "intent_type": interpretation.intent_type,
                },
            )
            async for event in graph.astream_events(
                {"messages": messages_for_graph},
                config={"configurable": {"dataset_id": dataset_id}},
                context=runtime_context,
                version="v2",
            ):
                event_name = event.get("event")
                event_name = str(event_name) if event_name is not None else ""
                data = event.get("data") or {}
                name = event.get("name")
                if event_name == "on_tool_start" and isinstance(name, str) and name in ML_DIRECT_TOOL_NAMES:
                    saw_ml_tool_call = True

                if event_name == "on_chain_stream":
                    continue

                if event_name == "on_chat_model_stream":
                    chunk = data.get("chunk") if isinstance(data, dict) else None
                    text = _extract_text_from_chunk(chunk)
                    text = _strip_internal_intent_payload_prefix(text)
                    if _looks_like_internal_intent_payload(text):
                        continue
                    if text:
                        if explicit_ml_request or chart_requested:
                            buffered_text_chunks.append(text)
                            artifact_type = _extract_structured_artifact_type(text)
                            if artifact_type in ML_RESULT_ARTIFACT_TYPES:
                                current_artifact = artifact_registry.get_latest(dataset_id, artifact_type=artifact_type)
                                prior_artifact_id = prior_model_artifact_ids.get(artifact_type)
                                current_artifact_id = current_artifact.get("artifact_id") if current_artifact else None
                                if current_artifact_id and current_artifact_id != prior_artifact_id:
                                    saw_ml_result = True
                                    produced_ml_artifact_types.add(str(artifact_type))
                        else:
                            yield _format_sse(
                                "message_chunk",
                                {
                                    "content": text,
                                    "dataset_id": dataset_id,
                                    "timestamp": _now_iso(),
                                },
                            )
                elif event_name == "on_tool_start":
                    yield _format_sse(
                        "tool_start",
                        {
                            "tool_name": name,
                            "dataset_id": dataset_id,
                            "timestamp": _now_iso(),
                        },
                    )
                elif event_name == "on_tool_end":
                    if isinstance(name, str) and name in ML_DIRECT_TOOL_NAMES:
                        for artifact_type in ML_RESULT_ARTIFACT_TYPES:
                            current_artifact = artifact_registry.get_latest(dataset_id, artifact_type=artifact_type)
                            prior_artifact_id = prior_model_artifact_ids.get(artifact_type)
                            if current_artifact is not None and current_artifact.get("artifact_id") != prior_artifact_id:
                                saw_ml_result = True
                                produced_ml_artifact_types.add(str(artifact_type))

                    tool_event = {
                        "tool_name": name,
                        "dataset_id": dataset_id,
                        "timestamp": _now_iso(),
                    }
                    yield _format_sse("tool_end", tool_event)

                    image_event = consume_current_image_event()
                    if image_event:
                        filename = image_event.get("filename")
                        if isinstance(filename, str) and filename:
                            yield _format_sse(
                                "image_generated",
                                {
                                    "type": "image_generated",
                                    "filename": filename,
                                    "image_url": _backend_image_url(request, filename),
                                    "tool_name": image_event.get("tool_name"),
                                    "dataset_id": dataset_id,
                                    "timestamp": _now_iso(),
                                },
                            )
                            saw_chart_image = True
                elif event_name == "on_chain_end":
                    continue

            if explicit_ml_request or chart_requested:
                if explicit_ml_request and not saw_ml_tool_call:
                    yield _format_sse(
                        "error",
                        {
                            "code": "structured_failure",
                            "message": "本次建模请求没有调用直接的 ml 工具，请先通过 ml_execute 完成建模。",
                        },
                    )
                    yield _format_sse("done", {"dataset_id": dataset_id, "timestamp": _now_iso()})
                    return

                required_ml_artifacts: set[str] = set()
                if ml_needs_training:
                    required_ml_artifacts.add("model_result")
                if ml_needs_metrics:
                    required_ml_artifacts.add("metrics_result")
                if ml_needs_feature_importance:
                    required_ml_artifacts.add("feature_importance_result")

                missing_ml_artifacts = required_ml_artifacts - produced_ml_artifact_types
                if explicit_ml_request and missing_ml_artifacts:
                    yield _format_sse(
                        "error",
                        {
                            "code": "structured_failure",
                            "message": f"本次建模请求缺少结构化结果：{', '.join(sorted(missing_ml_artifacts))}。",
                        },
                    )
                    yield _format_sse("done", {"dataset_id": dataset_id, "timestamp": _now_iso()})
                    return

                if chart_requested and not saw_chart_image:
                    yield _format_sse(
                        "error",
                        {
                            "code": "structured_failure",
                            "message": "本次图表请求没有成功生成可展示的图片结果，请检查字段名或图表描述后重试。",
                        },
                    )
                    yield _format_sse("done", {"dataset_id": dataset_id, "timestamp": _now_iso()})
                    return

                for text in buffered_text_chunks:
                    yield _format_sse(
                        "message_chunk",
                        {
                            "content": text,
                            "dataset_id": dataset_id,
                            "timestamp": _now_iso(),
                        },
                    )

            yield _format_sse("done", {"dataset_id": dataset_id, "timestamp": _now_iso()})
        except AppError as exc:
            yield _format_sse("error", {"code": exc.code, "message": exc.message})
            yield _format_sse("done", {"dataset_id": dataset_id, "timestamp": _now_iso()})
        except Exception:
            logger.exception("chat stream failed", extra={"dataset_id": dataset_id})
            yield _format_sse("error", {"code": "internal_error", "message": "服务器内部错误，请稍后重试。"})
            yield _format_sse("done", {"dataset_id": dataset_id, "timestamp": _now_iso()})
        finally:
            set_current_dataset_id(None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if IS_DEVELOPMENT:
    try:
        from langserve import add_routes
    except Exception:
        logger.warning("langserve 未安装或导入失败，开发环境将跳过 /agent 路由注入。")
    else:
        add_routes(
            app,
            graph,
            path="/agent",
            config_keys=["configurable"],
            playground_type="default",
        )


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    dataset_id: str | None = None
    body = b""
    should_rebuild = request.headers.get("content-type", "").startswith("application/json")

    if should_rebuild:
        body = await request.body()
        if body:
            try:
                payload = json.loads(body)
                if request.url.path.startswith("/agent") or request.url.path.startswith("/chat/stream"):
                    dataset_id = _extract_dataset_id_from_payload(payload)
                elif request.url.path.startswith("/calculate-correlation"):
                    if isinstance(payload, dict):
                        payload_dataset = payload.get("dataset_id")
                        if isinstance(payload_dataset, str):
                            dataset_id = payload_dataset
                elif request.url.path.startswith("/datasets/"):
                    dataset_id = request.url.path.rsplit("/", 1)[-1]
            except Exception:
                dataset_id = None

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive)

    if request.url.path.startswith("/agent"):
        set_current_dataset_id(dataset_id)

    response = await call_next(request)
    error_code = response.headers.get("X-Error-Code")
    logger.info(
        "%s %s status=%s dataset_id=%s error_code=%s",
        request.method,
        request.url.path,
        response.status_code,
        dataset_id,
        error_code,
    )

    if request.url.path.startswith("/agent"):
        set_current_dataset_id(None)

    return response


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8002"))
    reload_enabled = IS_DEVELOPMENT
    logger.info("Starting Data Agent backend", extra={"port": port, "images_dir": str(IMAGES_DIR)})
    uvicorn.run("src.server:app", host="0.0.0.0", port=port, reload=reload_enabled)
