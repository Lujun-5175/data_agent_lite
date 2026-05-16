from __future__ import annotations

import ast
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from io import StringIO
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from src.sse import extract_text_from_chunk

_PROXY_MARKERS = (
    "ReadOnlyDataFrameProxy",
    "ReadOnlySeriesProxy",
    "object at 0x",
)


def normalize_result_payload(value: Any, *, max_rows: int = 20) -> Any:
    """Return a JSON-safe representation for common tabular and scalar objects."""
    value = _unwrap_proxy(value)

    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [normalize_result_payload(item, max_rows=max_rows) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return _normalize_dataframe(value, max_rows=max_rows)
    if isinstance(value, pd.Series):
        return _normalize_series(value, max_rows=max_rows)
    if isinstance(value, pd.Index):
        return [normalize_result_payload(item, max_rows=max_rows) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): normalize_result_payload(item, max_rows=max_rows) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_result_payload(item, max_rows=max_rows) for item in value]
    if isinstance(value, set):
        return [normalize_result_payload(item, max_rows=max_rows) for item in sorted(value, key=lambda item: repr(item))]
    if is_dataclass(value):
        return normalize_result_payload(asdict(value), max_rows=max_rows)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict()
        except Exception:
            converted = None
        if isinstance(converted, (dict, list)):
            return normalize_result_payload(converted, max_rows=max_rows)

    return _fallback_string(value)


def summarize_tool_output(output: Any, *, max_rows: int = 20) -> dict[str, Any] | None:
    """Summarize tool output into JSON-safe structured payload and previews."""
    if output is None:
        return None

    if isinstance(output, str):
        cleaned_text = _clean_structured_text(output)
        payload = _summarize_text_output(output, max_rows=max_rows)
        if payload is None:
            return {
                "tool_output_preview": _truncate_text(cleaned_text or output, 400),
                "output_size_bytes": len(output.encode("utf-8")),
            }
        payload.setdefault("tool_output_preview", _truncate_text(cleaned_text or output, 400))
        payload.setdefault("output_size_bytes", len(output.encode("utf-8")))
        return payload

    if not isinstance(output, (dict, list, tuple, set, pd.DataFrame, pd.Series, pd.Index, np.ndarray, np.generic)):
        extracted_text = extract_text_from_chunk(output)
        if isinstance(extracted_text, str) and extracted_text.strip():
            return summarize_tool_output(extracted_text.strip(), max_rows=max_rows)

    normalized = normalize_result_payload(output, max_rows=max_rows)
    payload = _coerce_structured_payload(normalized, max_rows=max_rows)
    if payload is None:
        preview_text = _safe_preview_text(normalized)
        return {
            "tool_output_preview": _truncate_text(preview_text, 400),
            "output_size_bytes": len(preview_text.encode("utf-8")),
        }

    preview_text = _safe_preview_text(normalized)
    payload.setdefault("tool_output_preview", _truncate_text(preview_text, 400))
    payload.setdefault("output_size_bytes", len(preview_text.encode("utf-8")))
    return payload


def _normalize_dataframe(frame: pd.DataFrame, *, max_rows: int) -> dict[str, Any]:
    preview_frame = frame.head(max_rows).replace({np.nan: None})
    rows = [normalize_result_payload(row, max_rows=max_rows) for row in preview_frame.to_dict(orient="records")]
    return {
        "columns": [str(column) for column in frame.columns],
        "row_count": int(len(frame.index)),
        "rows": rows,
        "items": rows,
    }


def _normalize_series(series: pd.Series, *, max_rows: int) -> dict[str, Any]:
    preview_series = series.head(max_rows).replace({np.nan: None})
    rows = [
        {
            "index": normalize_result_payload(index, max_rows=max_rows),
            "value": normalize_result_payload(value, max_rows=max_rows),
        }
        for index, value in preview_series.items()
    ]
    payload: dict[str, Any] = {
        "columns": ["index", "value"],
        "row_count": int(len(series.index)),
        "rows": rows,
        "items": rows,
    }
    if series.name is not None:
        payload["name"] = normalize_result_payload(series.name, max_rows=max_rows)
    return payload


def _summarize_text_output(text: str, *, max_rows: int) -> dict[str, Any] | None:
    cleaned_text = _clean_structured_text(text)
    if not cleaned_text:
        return None

    parsed = _parse_structured_text(cleaned_text)
    if parsed is None:
        return None

    normalized = normalize_result_payload(parsed, max_rows=max_rows)
    return _coerce_structured_payload(normalized, max_rows=max_rows)


def _coerce_structured_payload(value: Any, *, max_rows: int) -> dict[str, Any] | None:
    if isinstance(value, dict):
        payload = {str(key): normalize_result_payload(item, max_rows=max_rows) for key, item in value.items()}
        if "rows" in payload and isinstance(payload["rows"], list):
            payload.setdefault("row_count", len(payload["rows"]))
        if "items" in payload and isinstance(payload["items"], list):
            payload.setdefault("row_count", len(payload["items"]))
        if "metrics" in payload and isinstance(payload["metrics"], dict):
            payload["metric_count"] = len(payload["metrics"])
        return payload

    if isinstance(value, list):
        items = [normalize_result_payload(item, max_rows=max_rows) for item in value[:max_rows]]
        payload: dict[str, Any] = {
            "items": items,
            "row_count": len(value),
        }
        if items and all(isinstance(item, dict) for item in items):
            payload["rows"] = items
        return payload

    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"value": value}

    return None


def _parse_structured_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
    except Exception:
        parsed = None
    if parsed is not None:
        return parsed

    try:
        parsed_literal = ast.literal_eval(stripped)
    except Exception:
        parsed_literal = None
    if isinstance(parsed_literal, (dict, list)):
        return parsed_literal

    table = _parse_fixed_width_table(stripped)
    if table is not None:
        return table
    return None


def _parse_fixed_width_table(text: str) -> list[dict[str, Any]] | None:
    cleaned_text = _clean_structured_text(text)
    if not cleaned_text:
        return None

    try:
        frame = pd.read_fwf(StringIO(cleaned_text))
    except Exception:
        return None
    if frame.empty:
        return None

    frame = frame.replace({np.nan: None})
    rows = frame.head(20).to_dict(orient="records")
    if not rows:
        return None

    filtered_rows = [row for row in rows if _row_is_meaningful(row)]
    if not filtered_rows:
        return None
    return filtered_rows


def _clean_structured_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "---":
            continue
        lowered = stripped.lower()
        if lowered.startswith("dtype:"):
            continue
        if lowered.startswith("name:") and "dtype:" in lowered:
            continue
        if lowered.startswith("length:") or lowered.startswith("freq:"):
            continue
        if any(marker.lower() in lowered for marker in _PROXY_MARKERS):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _row_is_meaningful(row: dict[str, Any]) -> bool:
    parts: list[str] = []
    for value in row.values():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    if not parts:
        return False

    joined = " ".join(parts).lower()
    if joined.startswith("dtype:"):
        return False
    if "readonlyseriesproxy" in joined or "readonlydataframeproxy" in joined:
        return False
    if "object at 0x" in joined:
        return False
    return True


def _unwrap_proxy(value: Any) -> Any:
    source = getattr(value, "_source", None)
    if isinstance(source, (pd.DataFrame, pd.Series)):
        return source.copy(deep=True)
    return value


def _safe_preview_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _truncate_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _fallback_string(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return repr(value)
