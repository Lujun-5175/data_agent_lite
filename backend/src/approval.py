from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.settings import SETTINGS


class ApprovalRisk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_type: str
    severity: str
    message: str
    metadata: dict[str, Any] | None = None


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str
    action_type: str
    proposed_action: str
    risks: list[ApprovalRisk] = Field(default_factory=list)
    requires_confirmation: bool
    metadata: dict[str, Any] | None = None


_LOW_CONFIDENCE_KEYS = ("confidence", "score", "intent_confidence", "route_confidence", "probability")
_JOIN_HINTS = ("join", "merge")
_DROP_ROWS_HINTS = ("drop rows", "drop_rows", "remove rows", "delete rows", "row drop")
_DESTRUCTIVE_HINTS = ("dropna", "drop_duplicates", "delete rows", "remove rows", "overwrite", "replace")
_IMPUTE_HINTS = ("fillna", "impute", "imputation", "fill missing", "missing value")
_OVERWRITE_HINTS = ("overwrite", "replace existing", "replace artifact", "overwrite artifact", "save as")
_TARGET_HINTS = ("target inferred", "infer target", "inferred target", "target_source", "target inferred")


def create_approval_request(
    action_type: str,
    proposed_action: str,
    risks: list[ApprovalRisk],
    metadata: dict[str, Any] | None = None,
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=uuid4().hex,
        action_type=action_type,
        proposed_action=proposed_action,
        risks=list(risks),
        requires_confirmation=requires_approval(risks),
        metadata=_json_safe_dict(metadata),
    )


def detect_risky_operation(operation: dict[str, Any] | str) -> list[ApprovalRisk]:
    if isinstance(operation, str):
        parsed = _maybe_parse_json(operation)
        if isinstance(parsed, dict):
            return detect_risky_operation(parsed)
        return _detect_from_text(operation)
    if not isinstance(operation, dict):
        return []

    risks: list[ApprovalRisk] = []
    text = " ".join(
        str(value)
        for value in (
            operation.get("action"),
            operation.get("operation"),
            operation.get("tool_name"),
            operation.get("description"),
        )
        if value is not None
    ).strip()
    lowered_text = text.casefold()

    if _has_join_inference(operation, lowered_text):
        risks.append(
            ApprovalRisk(
                risk_type="inferred_join_key",
                severity="medium",
                message="Join operation is using an inferred key and should be reviewed.",
                metadata=_risk_metadata(operation, "inferred_join_key"),
            )
        )

    if _has_drop_rows(operation, lowered_text):
        risks.append(
            ApprovalRisk(
                risk_type="drop_rows",
                severity="high",
                message="Rows may be dropped from the dataset.",
                metadata=_risk_metadata(operation, "drop_rows"),
            )
        )

    if _has_destructive_transform(operation, lowered_text):
        risks.append(
            ApprovalRisk(
                risk_type="destructive_transform",
                severity="high",
                message="The operation may destructively transform the dataset.",
                metadata=_risk_metadata(operation, "destructive_transform"),
            )
        )

    if _has_missing_value_imputation(operation, lowered_text):
        risks.append(
            ApprovalRisk(
                risk_type="missing_value_imputation",
                severity="medium",
                message="Missing values are being imputed or filled.",
                metadata=_risk_metadata(operation, "missing_value_imputation"),
            )
        )

    if _has_overwrite_artifact(operation, lowered_text):
        risks.append(
            ApprovalRisk(
                risk_type="overwrite_artifact",
                severity="high",
                message="The operation may overwrite an existing artifact.",
                metadata=_risk_metadata(operation, "overwrite_artifact"),
            )
        )

    if _has_model_target_inferred(operation, lowered_text):
        risks.append(
            ApprovalRisk(
                risk_type="model_target_inferred",
                severity="medium",
                message="The model target was inferred rather than explicitly provided.",
                metadata=_risk_metadata(operation, "model_target_inferred"),
            )
        )

    confidence = _extract_confidence(operation, lowered_text)
    if confidence is not None and confidence < SETTINGS.approval_low_confidence_threshold:
        risks.append(
            ApprovalRisk(
                risk_type="low_confidence_action",
                severity="medium",
                message=(
                    f"Action confidence {confidence:.2f} is below the approval threshold "
                    f"{SETTINGS.approval_low_confidence_threshold:.2f}."
                ),
                metadata={
                    "confidence": round(confidence, 6),
                    "threshold": SETTINGS.approval_low_confidence_threshold,
                },
            )
        )

    return _dedupe_risks(risks)


def requires_approval(risks: list[ApprovalRisk]) -> bool:
    return bool(risks)


def serialize_approval_request(request: ApprovalRequest) -> dict[str, Any]:
    return request.model_dump(mode="json", exclude_none=False)


def _detect_from_text(text: str) -> list[ApprovalRisk]:
    lowered = text.casefold()
    risks: list[ApprovalRisk] = []

    if any(token in lowered for token in _JOIN_HINTS) and ("infer" in lowered or "inferred" in lowered or "auto" in lowered):
        risks.append(
            ApprovalRisk(
                risk_type="inferred_join_key",
                severity="medium",
                message="Join operation appears to rely on an inferred key.",
                metadata={"source": "text"},
            )
        )

    if any(token in lowered for token in _DROP_ROWS_HINTS):
        risks.append(
            ApprovalRisk(
                risk_type="drop_rows",
                severity="high",
                message="The request appears to drop rows.",
                metadata={"source": "text"},
            )
        )

    if any(token in lowered for token in _DESTRUCTIVE_HINTS):
        risks.append(
            ApprovalRisk(
                risk_type="destructive_transform",
                severity="high",
                message="The request appears to make a destructive transform.",
                metadata={"source": "text"},
            )
        )

    if any(token in lowered for token in _IMPUTE_HINTS):
        risks.append(
            ApprovalRisk(
                risk_type="missing_value_imputation",
                severity="medium",
                message="The request appears to fill or impute missing values.",
                metadata={"source": "text"},
            )
        )

    if any(token in lowered for token in _OVERWRITE_HINTS):
        risks.append(
            ApprovalRisk(
                risk_type="overwrite_artifact",
                severity="high",
                message="The request may overwrite an existing artifact.",
                metadata={"source": "text"},
            )
        )

    if any(token in lowered for token in _TARGET_HINTS):
        risks.append(
            ApprovalRisk(
                risk_type="model_target_inferred",
                severity="medium",
                message="The model target appears to be inferred.",
                metadata={"source": "text"},
            )
        )

    confidence = _extract_confidence(text, lowered)
    if confidence is not None and confidence < SETTINGS.approval_low_confidence_threshold:
        risks.append(
            ApprovalRisk(
                risk_type="low_confidence_action",
                severity="medium",
                message=(
                    f"Action confidence {confidence:.2f} is below the approval threshold "
                    f"{SETTINGS.approval_low_confidence_threshold:.2f}."
                ),
                metadata={
                    "confidence": round(confidence, 6),
                    "threshold": SETTINGS.approval_low_confidence_threshold,
                },
            )
        )

    return _dedupe_risks(risks)


def _has_join_inference(operation: dict[str, Any], lowered_text: str) -> bool:
    inferred_flag = _truthy(operation.get("inferred_key")) or _truthy(operation.get("inferred_join_key"))
    key_source = _string_lower(operation.get("join_key_source") or operation.get("key_source") or operation.get("key_strategy"))
    if key_source in {"inferred", "auto", "automatic"}:
        inferred_flag = True
    if any(token in lowered_text for token in _JOIN_HINTS) and ("infer" in lowered_text or inferred_flag):
        return True
    return inferred_flag and any(token in lowered_text for token in _JOIN_HINTS)


def _has_drop_rows(operation: dict[str, Any], lowered_text: str) -> bool:
    action = _string_lower(operation.get("action") or operation.get("operation") or operation.get("name"))
    if action in {"drop_rows", "drop row", "delete_rows"}:
        return True
    if any(token in lowered_text for token in _DROP_ROWS_HINTS):
        return True
    return False


def _has_destructive_transform(operation: dict[str, Any], lowered_text: str) -> bool:
    action = _string_lower(operation.get("action") or operation.get("operation") or operation.get("name"))
    if action in {"dropna", "drop_duplicates", "overwrite", "replace"}:
        return True
    if any(token in lowered_text for token in _DESTRUCTIVE_HINTS):
        return True
    return False


def _has_missing_value_imputation(operation: dict[str, Any], lowered_text: str) -> bool:
    action = _string_lower(operation.get("action") or operation.get("operation") or operation.get("name"))
    if action in {"fillna", "impute", "imputation"}:
        return True
    if any(token in lowered_text for token in _IMPUTE_HINTS):
        return True
    return False


def _has_overwrite_artifact(operation: dict[str, Any], lowered_text: str) -> bool:
    if _truthy(operation.get("overwrite")) or _truthy(operation.get("replace")):
        return True
    mode = _string_lower(operation.get("mode"))
    if mode == "overwrite":
        return True
    artifact_id = operation.get("artifact_id")
    if artifact_id is not None and any(token in lowered_text for token in _OVERWRITE_HINTS):
        return True
    return any(token in lowered_text for token in _OVERWRITE_HINTS)


def _has_model_target_inferred(operation: dict[str, Any], lowered_text: str) -> bool:
    if _truthy(operation.get("target_inferred")):
        return True
    target_source = _string_lower(operation.get("target_source") or operation.get("target_strategy"))
    if target_source in {"inferred", "auto", "automatic"}:
        return True
    if any(token in lowered_text for token in _TARGET_HINTS):
        return True
    return False


def _extract_confidence(operation: dict[str, Any] | str, lowered_text: str) -> float | None:
    if isinstance(operation, dict):
        for key in _LOW_CONFIDENCE_KEYS:
            value = operation.get(key)
            confidence = _coerce_float(value)
            if confidence is not None:
                return confidence
        metadata = operation.get("metadata")
        if isinstance(metadata, dict):
            for key in _LOW_CONFIDENCE_KEYS:
                confidence = _coerce_float(metadata.get(key))
                if confidence is not None:
                    return confidence

    match = re.search(r"(?:confidence|score|probability)\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)", lowered_text)
    if match:
        return _coerce_float(match.group(1))
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _string_lower(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().casefold()
    return text or None


def _maybe_parse_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    if not stripped.startswith("{") and not stripped.startswith("["):
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _risk_metadata(operation: dict[str, Any], risk_type: str) -> dict[str, Any]:
    metadata = {
        "risk_type": risk_type,
        "operation": _json_safe_dict(operation),
    }
    return metadata


def _dedupe_risks(risks: list[ApprovalRisk]) -> list[ApprovalRisk]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[ApprovalRisk] = []
    for risk in risks:
        key = (risk.risk_type, risk.severity, risk.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(risk)
    return deduped


def _json_safe_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return {"value": _json_safe_value(value)}
    return {str(key): _json_safe_value(item) for key, item in value.items()}


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    return str(value)
