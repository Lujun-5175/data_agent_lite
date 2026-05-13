from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.runner import load_eval_cases, load_predictions, main
from src.eval.schema import EvalCase, EvalPrediction
from src.eval.metrics import score_predictions


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def test_loading_valid_jsonl_eval_cases(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    _write_jsonl(
        path,
        [
            {
                "case_id": "case-1",
                "category": "easy_stats",
                "user_query": "Summarize the numeric columns.",
            },
            {
                "case_id": "case-2",
                "category": "ml",
                "user_query": "Train a churn model.",
                "expected_tool": "ml_execute",
            },
        ],
    )

    cases = load_eval_cases(path)

    assert cases == [
        EvalCase(case_id="case-1", category="easy_stats", user_query="Summarize the numeric columns."),
        EvalCase(
            case_id="case-2",
            category="ml",
            user_query="Train a churn model.",
            expected_tool="ml_execute",
        ),
    ]


def test_loading_valid_jsonl_predictions(tmp_path: Path):
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "case_id": "case-1",
                "predicted_tool": "stats_execute",
                "execution_status": "success",
            },
            {
                "case_id": "case-2",
                "predicted_tool": "ml_execute",
                "execution_status": "error",
            },
        ],
    )

    predictions = load_predictions(path)

    assert predictions == [
        EvalPrediction(case_id="case-1", predicted_tool="stats_execute", execution_status="success"),
        EvalPrediction(case_id="case-2", predicted_tool="ml_execute", execution_status="error"),
    ]


def test_malformed_jsonl_line_raises_clear_value_error_with_line_number(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"case_id": "case-1", "category": "easy_stats", "user_query": "ok"}),
                "{not-json",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"line 2"):
        load_eval_cases(path)


def test_score_predictions_aligns_by_case_id_not_list_order():
    cases = [
        EvalCase(case_id="case-1", category="easy_stats", user_query="q1", expected_tool="stats_execute"),
        EvalCase(case_id="case-2", category="ml", user_query="q2", expected_tool="ml_execute"),
    ]
    predictions = [
        EvalPrediction(case_id="case-2", predicted_tool="ml_execute"),
        EvalPrediction(case_id="case-1", predicted_tool="stats_execute"),
    ]

    scores = score_predictions(cases, predictions)

    assert scores.tool_accuracy == pytest.approx(1.0)


def test_missing_prediction_does_not_crash_and_is_incorrect_where_applicable():
    cases = [
        EvalCase(
            case_id="case-1",
            category="adversarial_edge_cases",
            user_query="import os",
            expected_intent="unsafe_request",
            expected_tool="python_inter",
            expected_args={"py_code": "import os"},
            should_execute_successfully=False,
            should_be_blocked=True,
            expected_result={"blocked": True},
        )
    ]
    predictions: list[EvalPrediction] = []

    scores = score_predictions(cases, predictions)

    assert scores.num_cases == 1
    assert scores.intent_accuracy == 0.0
    assert scores.tool_accuracy == 0.0
    assert scores.argument_f1 == 0.0
    assert scores.execution_success_rate == 0.0
    assert scores.blocked_request_accuracy == 0.0
    assert scores.numerical_accuracy is None


def test_cli_scoring_runs_without_live_llm_calls(tmp_path: Path, capsys):
    cases_path = tmp_path / "cases.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        cases_path,
        [
            {
                "case_id": "case-1",
                "category": "easy_stats",
                "user_query": "Summarize the numeric columns.",
                "expected_tool": "stats_execute",
            }
        ],
    )
    _write_jsonl(
        predictions_path,
        [
            {
                "case_id": "case-1",
                "predicted_tool": "stats_execute",
            }
        ],
    )

    exit_code = main(["--cases", str(cases_path), "--predictions", str(predictions_path)])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    summary = json.loads(stdout)
    assert summary["num_cases"] == 1
    assert summary["tool_accuracy"] == pytest.approx(1.0)
