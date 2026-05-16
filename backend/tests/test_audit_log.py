from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src import audit_log, tools
from src.audit_log import AuditLogger, read_recent_records
from src.request_context import bind_request_context, set_route_diagnostics
from src.tools import SafeExecutionError

SCHEMA_FIELDS = [
    "run_id",
    "timestamp",
    "tool_name",
    "dataset_id",
    "tool_args",
    "code_sha256",
    "code_preview",
    "execution_status",
    "latency_ms",
    "output_size_bytes",
    "error_message",
    "blocked_reason",
    "extra",
]


def test_record_writes_valid_jsonl(tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=audit_path)

    logger.record(
        tool_name="stats_execute",
        dataset_id="dataset-1",
        tool_args={"action": "latest"},
        execution_status="success",
        latency_ms=1.2,
    )

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert list(record.keys()) == SCHEMA_FIELDS


def test_record_strips_code_to_hash_and_preview(tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    code = "x" * 500
    logger = AuditLogger(path=audit_path, max_code_chars=200)

    logger.record(
        tool_name="python_inter",
        dataset_id=None,
        tool_args={"py_code": None},
        code=code,
        execution_status="success",
        latency_ms=1.0,
    )

    record = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["code_sha256"] == hashlib.sha256(code.encode("utf-8")).hexdigest()
    assert record["code_preview"] == "x" * 200
    assert record["tool_args"] == {}
    assert code not in json.dumps(record, ensure_ascii=False)


def test_record_disabled_is_noop(tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(enabled=False, path=audit_path)

    run_id = logger.record(
        tool_name="python_inter",
        dataset_id=None,
        execution_status="success",
        latency_ms=0.1,
    )

    assert run_id
    assert not audit_path.exists() or audit_path.read_text(encoding="utf-8") == ""


def test_record_never_raises_on_write_failure(tmp_path: Path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=audit_path)

    def failing_open(self, *args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "open", failing_open)

    run_id = logger.record(
        tool_name="python_inter",
        dataset_id=None,
        execution_status="success",
        latency_ms=0.1,
    )

    assert run_id


def test_read_recent_records_returns_newest_first(tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=audit_path)
    run_ids = []
    for index in range(5):
        run_ids.append(
            logger.record(
                tool_name="stats_execute",
                dataset_id=f"dataset-{index}",
                tool_args={"index": index},
                execution_status="success",
                latency_ms=float(index),
            )
        )

    records = read_recent_records(path=audit_path, limit=3)

    assert [record["run_id"] for record in records] == list(reversed(run_ids[-3:]))


def test_read_recent_records_skips_malformed_lines(tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    valid_1 = {"run_id": "first"}
    valid_2 = {"run_id": "second"}
    audit_path.write_text(
        "\n".join(
            [
                json.dumps(valid_1),
                "{not-json",
                json.dumps(valid_2),
            ]
        ),
        encoding="utf-8",
    )

    records = read_recent_records(path=audit_path, limit=10)

    assert records == [valid_2, valid_1]


def test_record_status_blocked_includes_blocked_reason(tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=audit_path)

    logger.record(
        tool_name="python_inter",
        dataset_id="dataset-1",
        execution_status="blocked",
        latency_ms=2.0,
        error_message="bad code",
        blocked_reason="bad code",
    )

    record = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["execution_status"] == "blocked"
    assert record["blocked_reason"] == "bad code"


def test_relative_audit_path_resolves_from_project_root(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(audit_log, "PROJECT_ROOT", tmp_path)

    logger = AuditLogger(path="backend/audit_logs/audit.jsonl")

    assert logger.path == (tmp_path / "backend" / "audit_logs" / "audit.jsonl").resolve()


def test_stats_execute_safe_execution_error_audits_as_error(tmp_path: Path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=audit_path)

    class _FakeStats:
        def latest(self, artifact_type=None):
            raise SafeExecutionError("bad stats input")

    monkeypatch.setattr(tools, "get_audit_logger", lambda: logger)
    monkeypatch.setattr(tools, "_get_dataset_df", lambda: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(tools, "_build_helper_api", lambda df: (None, None, _FakeStats(), None, None))

    result = tools.stats_execute.func(action="latest")

    record = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert result == "错误：bad stats input"
    assert record["execution_status"] == "error"
    assert record["blocked_reason"] is None


def test_tool_audit_includes_route_diagnostics(tmp_path: Path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=audit_path)

    monkeypatch.setattr(tools, "get_audit_logger", lambda: logger)

    with bind_request_context("req-1"):
        set_route_diagnostics(
            {
                "final_intent": "analysis",
                "confidence": "low",
                "conflict_flags": ["ml_overcall"],
                "route_source": "llm_with_guardrail",
                "used_fallback": False,
            }
        )
        tools._record_tool_audit(
            tool_name="stats_execute",
            dataset_id="dataset-1",
            tool_args={"action": "latest"},
            code=None,
            execution_status="success",
            start=0.0,
            result="ok",
        )

    record = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["extra"]["routing"]["final_intent"] == "analysis"
    assert record["extra"]["routing"]["confidence"] == "low"
    assert record["extra"]["routing"]["conflict_flags"] == ["ml_overcall"]
