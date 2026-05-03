from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.agent import graph
from src.api_models import CorrelationRequest
from src.app_lifecycle import build_artifact_cleanup_lifespan
from src.chat_service import create_chat_stream_response
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
from src.request_parsing import error_response, extract_dataset_id_for_request_path
from src.settings import SETTINGS
from src.tools import bind_current_dataset_id

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
        return DEV_CORS_ORIGINS
    raise RuntimeError("CORS_ALLOW_ORIGINS must be configured when APP_ENV is production.")


app = FastAPI(
    title="Data Agent Backend",
    version="1.1",
    description="Data Agent Backend with safer dataset handling",
    lifespan=build_artifact_cleanup_lifespan(
        cleanup_func=cleanup_expired_artifacts,
        logger=logger,
        temp_dir_getter=lambda: TEMP_DATA_DIR,
        images_dir_getter=lambda: IMAGES_DIR,
        interval_seconds_getter=lambda: ARTIFACT_CLEANUP_INTERVAL_SECONDS,
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
        return JSONResponse(
            content={
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
        )
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
    return calculate_correlation(request.dataset_id, request.col1, request.col2)


@app.post("/chat/stream", response_model=None)
async def chat_stream(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise AppError("validation_error", "请求参数不合法，请检查后重试。", 422) from exc
    if not isinstance(payload, dict):
        raise AppError("validation_error", "请求参数不合法，请检查后重试。", 422)
    return await create_chat_stream_response(request, payload, graph=graph)


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
                if isinstance(payload, dict):
                    dataset_id = extract_dataset_id_for_request_path(request.url.path, payload)
            except Exception:
                dataset_id = None

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive)

    if request.url.path.startswith("/agent"):
        with bind_current_dataset_id(dataset_id):
            response = await call_next(request)
        response = _bind_agent_response_iterator(response, dataset_id)
    else:
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
    return response


def _bind_agent_response_iterator(response, dataset_id: str | None):
    body_iterator = getattr(response, "body_iterator", None)
    if body_iterator is None:
        return response

    async def _wrapped_iterator() -> AsyncIterator[bytes]:
        with bind_current_dataset_id(dataset_id):
            async for chunk in body_iterator:
                yield chunk

    response.body_iterator = _wrapped_iterator()
    return response


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8002"))
    logger.info("Starting Data Agent backend", extra={"port": port, "images_dir": str(IMAGES_DIR)})
    uvicorn.run("src.server:app", host="0.0.0.0", port=port, reload=IS_DEVELOPMENT)
