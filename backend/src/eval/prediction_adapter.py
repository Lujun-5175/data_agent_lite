from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.result_normalizer import normalize_result_payload

from .schema import EvalCase, EvalPrediction


def normalize_prediction(raw: dict[str, Any]) -> EvalPrediction:
    if not isinstance(raw, dict):
        raise TypeError("raw prediction must be a dictionary")

    payload = {
        "case_id": _coerce_required_str(raw.get("case_id") or raw.get("caseId"), field_name="case_id"),
        "predicted_intent": _coerce_code_str(
            _first_non_none(raw.get("predicted_intent"), raw.get("intent"), raw.get("intent_type"))
        ),
        "predicted_tool": _coerce_code_str(
            _first_non_none(raw.get("predicted_tool"), raw.get("tool_name"), raw.get("tool"))
        ),
        "predicted_args": _coerce_optional_dict(
            _first_non_none(raw.get("predicted_args"), raw.get("tool_args"), raw.get("args"))
        ),
        "execution_status": _coerce_code_str(
            _first_non_none(raw.get("execution_status"), raw.get("status"))
        ),
        "result": _normalize_optional_payload(_first_non_none(raw.get("result"), raw.get("payload"))),
        "error_message": _coerce_optional_str(_first_non_none(raw.get("error_message"), raw.get("error"))),
    }
    return EvalPrediction.model_validate(payload)


def write_predictions_jsonl(predictions: list[EvalPrediction], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [prediction.model_dump_json(exclude_none=False) for prediction in predictions]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def create_prediction_template(cases: list[EvalCase]) -> list[EvalPrediction]:
    return [
        EvalPrediction(
            case_id=case.case_id,
            predicted_intent=None,
            predicted_tool=None,
            predicted_args=None,
            execution_status=None,
            result=None,
            error_message=None,
        )
        for case in cases
    ]


def sanitize_tool_args(tool_args: Any) -> dict[str, Any] | None:
    if tool_args is None:
        return None
    if isinstance(tool_args, dict):
        sanitized: dict[str, Any] = {}
        for key, value in tool_args.items():
            if key == "py_code" and isinstance(value, str):
                sanitized[key] = _summarize_code(value)
            else:
                sanitized[key] = value
        return sanitized
    if isinstance(tool_args, str):
        stripped = tool_args.strip()
        if not stripped:
            return None
        return _summarize_code(stripped)
    return _coerce_optional_dict(tool_args)


def prediction_from_audit_record(record: dict[str, Any]) -> EvalPrediction | None:
    if not isinstance(record, dict):
        return None

    extra = record.get("extra")
    if not isinstance(extra, dict):
        return None

    case_id = extra.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        return None

    predicted_args = _coerce_optional_dict(record.get("tool_args"))
    result = _build_result_from_extra(extra)
    predicted_intent = _coerce_code_str(extra.get("predicted_intent"))

    try:
        return EvalPrediction(
            case_id=case_id.strip(),
            predicted_intent=predicted_intent,
            predicted_tool=_coerce_code_str(record.get("tool_name")),
            predicted_args=predicted_args,
            execution_status=_coerce_code_str(record.get("execution_status")),
            result=result,
            error_message=_coerce_optional_str(record.get("error_message")),
        )
    except Exception:
        return None


def _build_result_from_extra(extra: dict[str, Any]) -> dict[str, Any] | None:
    nested_result = extra.get("result")
    if isinstance(nested_result, dict):
        return normalize_result_payload(nested_result)

    result: dict[str, Any] = {}
    allowed_scalar_keys = {
        "artifact_type",
        "blocked_reason",
        "confidence_level",
        "group_by",
        "interpretation",
        "model_type",
        "note",
        "p_value",
        "row_count",
        "stats_type",
        "statistic",
        "test_type",
        "top_k",
    }
    allowed_collection_keys = {
        "items",
        "matrix",
        "metrics",
        "rows",
        "top_pairs",
        "warnings",
    }

    for key, value in extra.items():
        if key in {"case_id", "result", "tool_args"}:
            continue
        if key in allowed_scalar_keys and (isinstance(value, (str, int, float, bool)) or value is None):
            result[key] = normalize_result_payload(value)
        elif key in allowed_collection_keys and isinstance(value, (dict, list)):
            result[key] = normalize_result_payload(value)
    return result or None


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_required_str(value: Any, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    text = str(value).strip()
    if text:
        return text
    raise ValueError(f"{field_name} is required")


def _coerce_optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_code_str(value: Any) -> str | None:
    coerced = _coerce_optional_str(value)
    return coerced.lower() if coerced else None


def _coerce_optional_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            import json

            parsed = json.loads(stripped)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None
    return None


def _normalize_optional_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    normalized = normalize_result_payload(value)
    if isinstance(normalized, (dict, list)):
        if isinstance(normalized, list):
            payload: dict[str, Any] = {"items": normalized, "row_count": len(normalized)}
            if normalized and all(isinstance(item, dict) for item in normalized):
                payload["rows"] = normalized
            return payload
        return normalized
    return {"value": normalized}


def _summarize_code(code: str) -> dict[str, str]:
    return {
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "code_preview": code[:200],
    }
