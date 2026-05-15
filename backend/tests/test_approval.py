from __future__ import annotations

import json

from src.approval import (
    ApprovalRisk,
    create_approval_request,
    detect_risky_operation,
    requires_approval,
    serialize_approval_request,
)


def test_create_approval_request_has_approval_id():
    risks = [
        ApprovalRisk(
            risk_type="low_confidence_action",
            severity="medium",
            message="Action confidence is too low.",
            metadata={"confidence": 0.4},
        )
    ]

    request = create_approval_request(
        action_type="analysis",
        proposed_action="Train a model with inferred inputs",
        risks=risks,
        metadata={"source": "unit_test"},
    )

    assert request.approval_id
    assert request.action_type == "analysis"
    assert request.requires_confirmation is True


def test_inferred_join_key_triggers_approval():
    risks = detect_risky_operation({"action": "join", "inferred_key": True, "left": "orders", "right": "users"})

    assert any(risk.risk_type == "inferred_join_key" for risk in risks)
    assert requires_approval(risks) is True


def test_drop_rows_triggers_approval():
    risks = detect_risky_operation({"action": "drop_rows", "reason": "remove bad rows"})

    assert any(risk.risk_type == "drop_rows" for risk in risks)
    assert requires_approval(risks) is True


def test_fill_missing_triggers_approval():
    risks = detect_risky_operation({"action": "fillna", "strategy": "median"})

    assert any(risk.risk_type == "missing_value_imputation" for risk in risks)
    assert requires_approval(risks) is True


def test_low_confidence_triggers_approval():
    risks = detect_risky_operation({"action": "train", "confidence": 0.42})

    assert any(risk.risk_type == "low_confidence_action" for risk in risks)
    assert requires_approval(risks) is True


def test_safe_descriptive_operation_does_not_require_approval():
    risks = detect_risky_operation({"action": "describe_numeric", "columns": ["sales", "region"]})

    assert risks == []
    assert requires_approval(risks) is False


def test_serialization_is_json_safe():
    request = create_approval_request(
        action_type="analysis",
        proposed_action="Join two tables with inferred key",
        risks=detect_risky_operation({"action": "join", "inferred_key": True, "confidence": 0.2}),
        metadata={"nested": {"count": 1, "items": [1, 2, 3]}},
    )

    payload = serialize_approval_request(request)

    assert payload["approval_id"]
    assert payload["requires_confirmation"] is True
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
