from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.settings import SETTINGS

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

AuditStatus = Literal["success", "error", "timeout", "blocked"]


def _resolve_audit_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


class AuditLogger:
    def __init__(
        self,
        *,
        enabled: bool = True,
        path: str | Path = "backend/audit_logs/audit.jsonl",
        max_code_chars: int = 200,
    ) -> None:
        self.enabled = enabled
        self.path = _resolve_audit_path(path)
        self.max_code_chars = max_code_chars

    def record(
        self,
        *,
        tool_name: str,
        dataset_id: str | None,
        tool_args: dict[str, Any] | None = None,
        code: str | None = None,
        execution_status: AuditStatus,
        latency_ms: float,
        output_size_bytes: int = 0,
        error_message: str | None = None,
        blocked_reason: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        run_id = uuid.uuid4().hex
        if not self.enabled:
            return run_id

        try:
            filtered_tool_args = {key: value for key, value in (tool_args or {}).items() if value is not None}
            code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest() if code is not None else None
            code_preview = code[: self.max_code_chars] if code is not None else None
            record = {
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool_name": tool_name,
                "dataset_id": dataset_id,
                "tool_args": filtered_tool_args,
                "code_sha256": code_sha256,
                "code_preview": code_preview,
                "execution_status": execution_status,
                "latency_ms": latency_ms,
                "output_size_bytes": output_size_bytes,
                "error_message": error_message,
                "blocked_reason": blocked_reason,
                "extra": extra,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as target:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("Failed to write audit log record", exc_info=True)
        return run_id


_AUDIT_LOGGER: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    global _AUDIT_LOGGER
    if _AUDIT_LOGGER is None:
        _AUDIT_LOGGER = AuditLogger(
            enabled=SETTINGS.audit_log_enabled,
            path=SETTINGS.audit_log_path,
            max_code_chars=SETTINGS.audit_log_max_code_chars,
        )
    return _AUDIT_LOGGER


def read_recent_records(
    *,
    path: str | Path | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    audit_path = _resolve_audit_path(path or SETTINGS.audit_log_path)
    if not audit_path.exists():
        return []

    try:
        records: list[dict[str, Any]] = []
        with audit_path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    records.append(parsed)
        return list(reversed(records))[:limit]
    except Exception:
        logger.warning("Failed to read audit log records", exc_info=True)
        return []
