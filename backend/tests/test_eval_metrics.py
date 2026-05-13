from __future__ import annotations

import hashlib

import pytest

from src.eval.metrics import (
    argument_f1,
    blocked_request_accuracy,
    execution_success_rate,
    intent_accuracy,
    numerical_accuracy,
    score_predictions,
    tool_accuracy,
)
from src.eval.schema import EvalCase, EvalPrediction


def _case(case_id: str, **kwargs) -> EvalCase:
    return EvalCase(case_id=case_id, category="test", user_query=f"query {case_id}", **kwargs)


def _prediction(case_id: str, **kwargs) -> EvalPrediction:
    return EvalPrediction(case_id=case_id, **kwargs)


def test_intent_accuracy_calculation():
    cases = [
        _case("case-1", expected_intent="stats"),
        _case("case-2", expected_intent="ml"),
    ]
    predictions = [
        _prediction("case-1", predicted_intent="stats"),
        _prediction("case-2", predicted_intent="python"),
    ]

    assert intent_accuracy(cases, predictions) == pytest.approx(0.5)


def test_tool_accuracy_calculation():
    cases = [
        _case("case-1", expected_tool="stats_execute"),
        _case("case-2", expected_tool="ml_execute"),
    ]
    predictions = [
        _prediction("case-1", predicted_tool="stats_execute"),
        _prediction("case-2", predicted_tool="python_inter"),
    ]

    assert tool_accuracy(cases, predictions) == pytest.approx(0.5)


def test_tool_accuracy_exact_match_still_works():
    cases = [_case("case-1", expected_tool="stats_execute")]
    predictions = [_prediction("case-1", predicted_tool="stats_execute")]

    assert tool_accuracy(cases, predictions) == pytest.approx(1.0)


def test_tool_accuracy_accepts_accepted_tools():
    cases = [_case("case-1", expected_tool="stats_execute", accepted_tools=["stats_execute", "python_inter"])]
    predictions = [_prediction("case-1", predicted_tool="python_inter")]

    assert tool_accuracy(cases, predictions) == pytest.approx(1.0)


def test_argument_f1_exact_match():
    cases = [_case("case-1", expected_args={"a": 1, "b": 2})]
    predictions = [_prediction("case-1", predicted_args={"a": 1, "b": 2})]

    assert argument_f1(cases, predictions) == pytest.approx(1.0)


def test_argument_f1_partial_match():
    cases = [_case("case-1", expected_args={"a": 1, "b": 2})]
    predictions = [_prediction("case-1", predicted_args={"a": 1, "b": 3})]

    assert argument_f1(cases, predictions) == pytest.approx(0.5)


def test_argument_f1_list_values_match_as_set():
    cases = [_case("case-1", expected_args={"cols": ["a", "b"]})]
    predictions = [_prediction("case-1", predicted_args={"cols": ["b", "a"]})]

    assert argument_f1(cases, predictions) == pytest.approx(1.0)


def test_argument_f1_nested_dict_values_match():
    cases = [_case("case-1", expected_args={"outer": {"a": 1, "b": 2}})]
    predictions = [_prediction("case-1", predicted_args={"outer": {"a": 1, "b": 3}})]

    assert argument_f1(cases, predictions) == pytest.approx(0.5)


def test_argument_f1_ignores_code_fulltext_when_sanitized():
    code = "print('hello')"
    cases = [_case("case-1", expected_args={"py_code": code})]
    predictions = [
        _prediction(
            "case-1",
            predicted_args={
                "py_code": {
                    "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                    "code_preview": code,
                }
            },
        )
    ]

    assert argument_f1(cases, predictions) == pytest.approx(1.0)


def test_argument_f1_ignores_dynamic_artifact_ids():
    cases = [
        _case(
            "case-1",
            expected_args={
                "action": "metrics",
                "model_artifact_id": "telco_churn_churn_model",
                "top_k": 5,
            },
        )
    ]
    predictions = [
        _prediction(
            "case-1",
            predicted_args={
                "action": "metrics",
                "model_artifact_id": "6a5d8a3d-7ed3-4a2a-9d46-8d92c2a7eaa9",
                "top_k": 5,
            },
        )
    ]

    assert argument_f1(cases, predictions) == pytest.approx(1.0)


def test_execution_success_rate():
    cases = [
        _case("case-1", should_execute_successfully=True),
        _case("case-2", should_execute_successfully=False),
    ]
    predictions = [
        _prediction("case-1", execution_status="success"),
        _prediction("case-2", execution_status="error"),
    ]

    assert execution_success_rate(cases, predictions) == pytest.approx(1.0)


def test_blocked_request_accuracy():
    cases = [
        _case("case-1", should_be_blocked=True),
        _case("case-2", should_be_blocked=False),
    ]
    predictions = [
        _prediction("case-1", execution_status="blocked"),
        _prediction("case-2", execution_status="success"),
    ]

    assert blocked_request_accuracy(cases, predictions) == pytest.approx(1.0)


def test_numerical_accuracy_relative_tolerance():
    cases = [_case("case-1", expected_result={"value": 100.0}, stable_numerical_result=True)]
    predictions = [_prediction("case-1", result={"value": 100.05})]

    assert numerical_accuracy(cases, predictions) == pytest.approx(1.0)


def test_numerical_accuracy_zero_expected_absolute_tolerance():
    cases = [_case("case-1", expected_result={"value": 0.0}, stable_numerical_result=True)]
    predictions = [_prediction("case-1", result={"value": 5e-7})]

    assert numerical_accuracy(cases, predictions) == pytest.approx(1.0)


def test_numerical_accuracy_skips_unstable_cases():
    cases = [_case("case-1", expected_result={"value": 100.0}, stable_numerical_result=False)]
    predictions = [_prediction("case-1", result={"value": 100.0})]

    assert numerical_accuracy(cases, predictions) is None


def test_numerical_accuracy_counts_stable_cases():
    cases = [
        _case("case-1", expected_result={"missing_count": 11}, stable_numerical_result=True),
        _case("case-2", expected_result={"row_count": 4}, stable_numerical_result=True),
    ]
    predictions = [
        _prediction("case-1", result={"rows": [{"missing_count": 11}]}),
        _prediction("case-2", result={"row_count": 4}),
    ]

    assert numerical_accuracy(cases, predictions) == pytest.approx(1.0)


def test_metrics_return_none_when_no_comparable_cases():
    cases = [_case("case-1")]
    predictions = [_prediction("case-1")]

    scores = score_predictions(cases, predictions)

    assert scores.num_cases == 1
    assert scores.intent_accuracy is None
    assert scores.tool_accuracy is None
    assert scores.argument_f1 is None
    assert scores.execution_success_rate is None
    assert scores.blocked_request_accuracy is None
    assert scores.numerical_accuracy is None
