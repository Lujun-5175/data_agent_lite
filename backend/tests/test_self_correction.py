from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import tools
from src.audit_log import AuditLogger
from src.self_correction import (
    RepairSuggestion,
    StructuredExecutionError,
    build_repair_prompt,
    classify_execution_error,
    extract_missing_column,
    should_retry_error,
    suggest_similar_columns,
)
from src.tools import SafeExecutionError, ToolExecutionTimeoutError


def test_extract_missing_column_from_keyerror():
    assert extract_missing_column("KeyError: 'salse'") == "salse"


def test_extract_missing_column_from_not_in_index():
    assert extract_missing_column("['salse'] not in index") == "salse"


def test_suggest_similar_columns_returns_best_match():
    suggestions = suggest_similar_columns("salse", ["sales", "date", "region"])

    assert suggestions
    assert suggestions[0].suggestion == "sales"


def test_suggest_similar_columns_ignores_weak_matches():
    suggestions = suggest_similar_columns("qzxy", ["sales", "date", "region"])

    assert suggestions == []


def test_classify_missing_column_retryable():
    structured = classify_execution_error("KeyError: 'salse'", available_columns=["sales", "date", "region"])

    assert structured.error_type == "missing_column"
    assert structured.retryable is True
    assert structured.safe_to_retry is True
    assert any(suggestion.suggestion == "sales" for suggestion in structured.suggestions)


def test_classify_safe_execution_error_not_retryable():
    structured = classify_execution_error(SafeExecutionError("bad code"))

    assert structured.error_type == "safe_execution_blocked"
    assert structured.retryable is False
    assert structured.safe_to_retry is False


def test_classify_timeout_not_retryable():
    structured = classify_execution_error(ToolExecutionTimeoutError(1.0))

    assert structured.error_type == "timeout"
    assert structured.retryable is False
    assert structured.safe_to_retry is False


def test_classify_syntax_error_retryable():
    structured = classify_execution_error(SyntaxError("invalid syntax"))

    assert structured.error_type == "syntax_error"
    assert structured.retryable is True
    assert structured.safe_to_retry is True


def test_classify_name_error_retryable():
    structured = classify_execution_error(NameError("name 'sales' is not defined"), available_columns=["sales"])

    assert structured.error_type == "name_error"
    assert structured.retryable is True
    assert structured.safe_to_retry is True


def test_build_repair_prompt_contains_suggestions():
    structured = StructuredExecutionError(
        error_type="missing_column",
        raw_message="KeyError: 'salse'",
        retryable=True,
        missing_column="salse",
        suggestions=[
            RepairSuggestion(original="salse", suggestion="sales", score=0.91, reason="列名字符串相似"),
        ],
        safe_to_retry=True,
    )

    prompt = build_repair_prompt(
        original_code="print(df['salse'])",
        structured_error=structured,
        available_columns=["sales", "date", "region"],
    )

    assert "missing_column: salse" in prompt
    assert "sales" in prompt
    assert "available_columns: sales, date, region" in prompt
    assert "不要使用 import、文件访问、eval、exec 或任何被禁止的 API。" in prompt
    assert "print(df['salse'])" in prompt


def test_should_retry_respects_attempt_budget():
    structured = StructuredExecutionError(
        error_type="name_error",
        raw_message="name 'sales' is not defined",
        retryable=True,
        safe_to_retry=True,
    )

    assert should_retry_error(structured, attempt=0, max_attempts=2) is True
    assert should_retry_error(structured, attempt=2, max_attempts=2) is False

    blocked = StructuredExecutionError(
        error_type="safe_execution_blocked",
        raw_message="bad code",
        retryable=False,
        safe_to_retry=False,
    )
    assert should_retry_error(blocked, attempt=0, max_attempts=2) is False


def test_structured_error_json_serializable():
    structured = StructuredExecutionError(
        error_type="missing_column",
        raw_message="KeyError: 'salse'",
        retryable=True,
        missing_column="salse",
        suggestions=[
            RepairSuggestion(original="salse", suggestion="sales", score=0.91, reason="列名字符串相似"),
        ],
        safe_to_retry=True,
    )

    payload = structured.model_dump()
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


def test_python_inter_error_audit_extra_contains_structured_error(tmp_path: Path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=audit_path)
    df = pd.DataFrame({"sales": [1, 2, 3], "date": ["2024-01-01"] * 3, "region": ["east", "west", "east"]})

    monkeypatch.setattr(tools, "get_audit_logger", lambda: logger)
    monkeypatch.setattr(tools, "_get_dataset_df", lambda: df)

    result = tools.python_inter.func(py_code="print(df['salse'])")

    assert "代码执行失败" in result
    record = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    structured_error = record["extra"]["structured_error"]
    assert structured_error["error_type"] == "missing_column"
    assert structured_error["retryable"] is True
    assert structured_error["safe_to_retry"] is True
    assert structured_error["missing_column"] == "salse"
    assert structured_error["suggestions"][0]["suggestion"] == "sales"
