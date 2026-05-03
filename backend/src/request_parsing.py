from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

from src.api_models import ErrorPayload


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": ErrorPayload(code=code, message=message).model_dump()},
        headers={"X-Error-Code": code},
    )


def extract_dataset_id_from_payload(payload: dict[str, object]) -> str | None:
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


def extract_messages(payload: dict[str, object]) -> list[dict[str, object]]:
    input_payload = payload.get("input")
    if isinstance(input_payload, dict):
        messages = input_payload.get("messages")
        if isinstance(messages, list):
            return [message for message in messages if isinstance(message, dict)]
    messages = payload.get("messages")
    if isinstance(messages, list):
        return [message for message in messages if isinstance(message, dict)]
    return []


def extract_latest_user_message(messages: list[dict[str, object]]) -> str:
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


def has_prior_analysis_context(messages: list[dict[str, object]]) -> bool:
    for message in messages[:-1]:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        normalized = content.lower()
        if any(token in normalized for token in ("dataset", "数据", "统计", "相关性", "group", "检验", "analysis")):
            return True
    return False


def extract_dataset_id_for_request_path(path: str, payload: dict[str, Any]) -> str | None:
    if path.startswith("/agent") or path.startswith("/chat/stream"):
        return extract_dataset_id_from_payload(payload)
    if path.startswith("/calculate-correlation"):
        payload_dataset = payload.get("dataset_id")
        if isinstance(payload_dataset, str) and payload_dataset.strip():
            return payload_dataset.strip()
    if path.startswith("/datasets/"):
        return path.rsplit("/", 1)[-1]
    return None
