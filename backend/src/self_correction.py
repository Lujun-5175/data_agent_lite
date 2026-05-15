from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RepairSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original: str
    suggestion: str
    score: float
    reason: str | None = None


class StructuredExecutionError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_type: str
    raw_message: str
    retryable: bool
    missing_column: str | None = None
    suggestions: list[RepairSuggestion] = Field(default_factory=list)
    safe_to_retry: bool = False
    repair_prompt: str | None = None


_TOOL_PREFIX_RE = re.compile(r"^(?:代码执行失败|绘图执行失败|错误)\s*[:：]\s*", re.IGNORECASE)
_QUOTED_COLUMN_RE = re.compile(r"^\s*['\"](?P<column>[^'\"]+)['\"]\s*$")
_KEYERROR_RE = re.compile(r"KeyError\s*[:(]\s*['\"](?P<column>[^'\"]+)['\"]\s*\)?", re.IGNORECASE)
_KEYERROR_CALL_RE = re.compile(r"KeyError\((?P<quote>['\"])(?P<column>[^'\"]+)(?P=quote)\)", re.IGNORECASE)
_NOT_IN_INDEX_RE = re.compile(r"\[\s*['\"](?P<column>[^'\"]+)['\"]\s*\]\s+not in index", re.IGNORECASE)
_NOT_IN_INDEX_QUOTED_RE = re.compile(r"['\"](?P<column>[^'\"]+)['\"]\s+not in index", re.IGNORECASE)
_COLUMN_NOT_FOUND_RE = re.compile(r"(?:列不存在|排序列不存在)\s*[:：]\s*['\"]?(?P<column>[^'\"\n\r]+?)['\"]?(?:\s|$)", re.IGNORECASE)
_NAME_ERROR_RE = re.compile(r"name\s+['\"](?P<name>[^'\"]+)['\"]\s+is\s+not\s+defined", re.IGNORECASE)


def extract_missing_column(error_message: str) -> str | None:
    message = error_message.strip()
    if not message:
        return None

    candidates = [
        _KEYERROR_RE.search(message),
        _KEYERROR_CALL_RE.search(message),
        _NOT_IN_INDEX_RE.search(message),
        _NOT_IN_INDEX_QUOTED_RE.search(message),
        _COLUMN_NOT_FOUND_RE.search(message),
        _QUOTED_COLUMN_RE.search(message),
    ]
    for match in candidates:
        if not match:
            continue
        column = match.group("column").strip()
        if column:
            return column
    return None


def suggest_similar_columns(
    missing_column: str,
    available_columns: list[str],
    *,
    max_suggestions: int = 3,
) -> list[RepairSuggestion]:
    if max_suggestions <= 0 or not available_columns:
        return []

    normalized_missing = _normalize_token(missing_column)
    if not normalized_missing:
        return []

    scored: list[RepairSuggestion] = []
    seen: set[str] = set()
    for column in available_columns:
        candidate = str(column).strip()
        if not candidate:
            continue
        normalized_candidate = _normalize_token(candidate)
        if not normalized_candidate or normalized_candidate in seen:
            continue
        seen.add(normalized_candidate)
        score = SequenceMatcher(None, normalized_missing, normalized_candidate).ratio()
        if score < 0.6:
            continue
        reason = "列名字符串相似"
        scored.append(
            RepairSuggestion(
                original=missing_column,
                suggestion=candidate,
                score=round(score, 6),
                reason=reason,
            )
        )

    scored.sort(key=lambda item: (-item.score, item.suggestion.lower()))
    return scored[:max_suggestions]


def classify_execution_error(
    error: BaseException | str,
    *,
    available_columns: list[str] | None = None,
) -> StructuredExecutionError:
    raw_message = _coerce_message(error)
    normalized_message = _strip_tool_prefixes(raw_message)
    class_name = error.__class__.__name__ if isinstance(error, BaseException) else None

    missing_column = extract_missing_column(normalized_message)
    suggestions: list[RepairSuggestion] = []
    retryable = False
    safe_to_retry = False
    error_type = "generic_error"

    if _is_timeout_error(class_name, normalized_message):
        error_type = "timeout"
    elif _looks_like_syntax_error(class_name, normalized_message):
        error_type = "syntax_error"
        retryable = True
        safe_to_retry = True
    elif _looks_like_name_error(class_name, normalized_message):
        error_type = "name_error"
        retryable = True
        safe_to_retry = True
        if available_columns:
            name_hint = _extract_name_hint(normalized_message)
            if name_hint:
                suggestions = suggest_similar_columns(name_hint, available_columns)
    elif missing_column is not None:
        error_type = "missing_column"
        retryable = True
        if available_columns:
            suggestions = suggest_similar_columns(missing_column, available_columns)
            safe_to_retry = bool(suggestions)
    elif _looks_like_safe_execution_blocked(class_name, normalized_message):
        error_type = "safe_execution_blocked"
    else:
        error_type = "generic_error"

    structured = StructuredExecutionError(
        error_type=error_type,
        raw_message=raw_message,
        retryable=retryable,
        missing_column=missing_column,
        suggestions=suggestions,
        safe_to_retry=safe_to_retry,
    )
    structured.repair_prompt = build_repair_prompt(
        original_code=None,
        structured_error=structured,
        available_columns=available_columns,
    )
    return structured


def build_repair_prompt(
    *,
    original_code: str | None,
    structured_error: StructuredExecutionError,
    available_columns: list[str] | None = None,
) -> str:
    lines = [
        "请修复上一次执行失败的代码，尽量只做最小改动。",
        f"error_type: {structured_error.error_type}",
        f"raw_message: {structured_error.raw_message}",
    ]
    if structured_error.missing_column:
        lines.append(f"missing_column: {structured_error.missing_column}")
    if structured_error.suggestions:
        suggestion_text = ", ".join(
            f"{item.suggestion} ({item.score:.2f})" for item in structured_error.suggestions
        )
        lines.append(f"suggested_columns: {suggestion_text}")
    if available_columns:
        lines.append(f"available_columns: {', '.join(str(column) for column in available_columns)}")
    lines.append("不要使用 import、文件访问、eval、exec 或任何被禁止的 API。")
    if original_code:
        lines.append("original_code:")
        lines.append("```python")
        lines.append(original_code)
        lines.append("```")
    return "\n".join(lines)


def should_retry_error(
    structured_error: StructuredExecutionError,
    attempt: int,
    max_attempts: int,
) -> bool:
    return bool(structured_error.retryable and structured_error.safe_to_retry and attempt < max_attempts)


def _coerce_message(error: BaseException | str) -> str:
    if isinstance(error, BaseException):
        return str(error)
    return str(error)


def _strip_tool_prefixes(message: str) -> str:
    stripped = message.strip()
    while True:
        updated = _TOOL_PREFIX_RE.sub("", stripped, count=1)
        if updated == stripped:
            return stripped
        stripped = updated.strip()


def _normalize_token(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _is_timeout_error(class_name: str | None, message: str) -> bool:
    if class_name == "ToolExecutionTimeoutError":
        return True
    lower_message = message.casefold()
    return "超时" in lower_message or "timeout" in lower_message


def _looks_like_syntax_error(class_name: str | None, message: str) -> bool:
    lower_message = message.casefold()
    if class_name == "SyntaxError":
        return True
    return lower_message.startswith("python 语法错误") or "invalid syntax" in lower_message or "syntaxerror" in lower_message


def _looks_like_name_error(class_name: str | None, message: str) -> bool:
    if class_name == "NameError":
        return True
    return _NAME_ERROR_RE.search(message) is not None or "nameerror" in message.casefold()


def _extract_name_hint(message: str) -> str | None:
    match = _NAME_ERROR_RE.search(message)
    if not match:
        return None
    name = match.group("name").strip()
    return name or None


def _looks_like_safe_execution_blocked(class_name: str | None, message: str) -> bool:
    if class_name == "SafeExecutionError":
        return True
    lower_message = message.casefold()
    return any(
        needle in lower_message
        for needle in (
            "安全策略拦截",
            "不允许",
            "危险",
            "敏感",
            "file access is forbidden",
            "import statements are forbidden",
            "forbidden",
        )
    )
