from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import Request
from fastapi.responses import StreamingResponse


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_sse(event_type: str, payload: dict[str, object]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def extract_text_from_chunk(chunk: object) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        content = chunk.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(extract_text_from_chunk(item) for item in content)
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(extract_text_from_chunk(item) for item in content)
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    return ""


def backend_image_url(request: Request, filename: str) -> str:
    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/static/images/{filename}"


def build_streaming_response(event_generator: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
