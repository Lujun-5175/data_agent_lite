from __future__ import annotations

import json
from pathlib import Path

from src.eval.prediction_adapter import (
    create_prediction_template,
    normalize_prediction,
    prediction_from_audit_record,
    write_predictions_jsonl,
)
from src.eval.runner import load_predictions
from src.eval.schema import EvalCase, EvalPrediction


def test_normalize_prediction_valid_dict():
    raw = {
        "case_id": "case-1",
        "predicted_intent": "Stats",
        "tool_name": "STATS_EXECUTE",
        "tool_args": {"action": "latest"},
        "status": "SUCCESS",
        "result": {"row_count": 3},
        "error": None,
    }

    prediction = normalize_prediction(raw)

    assert prediction == EvalPrediction(
        case_id="case-1",
        predicted_intent="stats",
        predicted_tool="stats_execute",
        predicted_args={"action": "latest"},
        execution_status="success",
        result={"row_count": 3},
        error_message=None,
    )


def test_create_prediction_template_matches_cases():
    cases = [
        EvalCase(case_id="case-1", category="easy_stats", user_query="q1"),
        EvalCase(case_id="case-2", category="ml", user_query="q2"),
    ]

    template = create_prediction_template(cases)

    assert template == [
        EvalPrediction(case_id="case-1"),
        EvalPrediction(case_id="case-2"),
    ]


def test_write_predictions_jsonl_roundtrip(tmp_path: Path):
    predictions = [
        EvalPrediction(case_id="case-1", predicted_tool="stats_execute", execution_status="success"),
        EvalPrediction(case_id="case-2", predicted_tool="ml_execute", execution_status="error"),
    ]
    path = tmp_path / "predictions.jsonl"

    write_predictions_jsonl(predictions, path)
    loaded = load_predictions(path)

    assert loaded == predictions


def test_prediction_from_audit_record_with_case_id():
    record = {
        "tool_name": "stats_execute",
        "tool_args": {"action": "group_summary"},
        "execution_status": "success",
        "error_message": None,
        "extra": {
            "case_id": "case-1",
            "stats_type": "group_summary",
            "row_count": 4,
        },
    }

    prediction = prediction_from_audit_record(record)

    assert prediction == EvalPrediction(
        case_id="case-1",
        predicted_tool="stats_execute",
        predicted_args={"action": "group_summary"},
        execution_status="success",
        result={"stats_type": "group_summary", "row_count": 4},
        error_message=None,
    )


def test_prediction_from_audit_record_without_case_id_returns_none():
    record = {
        "tool_name": "stats_execute",
        "tool_args": {"action": "latest"},
        "execution_status": "success",
        "extra": {"note": "missing case id"},
    }

    assert prediction_from_audit_record(record) is None


def test_prediction_from_malformed_audit_record_does_not_crash():
    record = {
        "tool_name": object(),
        "execution_status": 123,
        "extra": None,
    }

    assert prediction_from_audit_record(record) is None

