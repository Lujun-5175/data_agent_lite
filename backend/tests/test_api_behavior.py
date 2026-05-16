from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from src import server
from src import chat_service
from src.audit_log import AuditLogger
from fastapi.testclient import TestClient


def test_chat_stream_requires_dataset_for_dataset_specific_prompt(client: TestClient):
    response = client.post(
        "/chat/stream",
        json={
            "input": {
                "messages": [
                    {
                        "type": "human",
                        "content": "请根据这份数据做相关性分析",
                    }
                ]
            }
        },
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "dataset_required"
    assert "请先上传 CSV 文件" in payload["error"]["message"]


def test_error_payload_shape_stable(client: TestClient):
    response = client.post(
        "/upload",
        files={"file": ("invalid.txt", b"not csv", "text/plain")},
    )
    assert response.status_code == 400
    payload = response.json()
    assert set(payload.keys()) == {"error", "request_id"}
    assert set(payload["error"].keys()) == {"code", "message"}
    assert payload["error"]["code"] == "invalid_file_type"
    assert payload["request_id"]


def test_correlation_invalid_columns_returns_safe_error(client: TestClient):
    upload_response = client.post(
        "/upload",
        files={"file": ("corr.csv", b"a,b\n1,2\n3,4\n", "text/csv")},
    )
    assert upload_response.status_code == 200
    dataset_id = upload_response.json()["dataset_id"]

    response = client.post(
        "/calculate-correlation",
        json={"dataset_id": dataset_id, "col1": "a", "col2": "missing_col"},
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "dataset_load_error"


def test_dataset_overview_request_streams_metadata_without_agent_loop(client: TestClient):
    upload_response = client.post(
        "/upload",
        files={
            "file": (
                "sales_sample.csv",
                (
                    b"order_date,total_amount,product_category,region,channel\n"
                    b"2025-01-01,120.5,Electronics,West,Online\n"
                    b"2025-01-02,88.0,Home,East,Offline\n"
                ),
                "text/csv",
            )
        },
    )
    assert upload_response.status_code == 200
    dataset_id = upload_response.json()["dataset_id"]

    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "input": {
                "messages": [
                    {
                        "type": "human",
                        "content": "讲解数据集",
                    }
                ]
            },
            "config": {"configurable": {"dataset_id": dataset_id}},
        },
    )

    assert response.status_code == 200
    assert "event: message_chunk" in response.text
    assert "sales_sample.csv" in response.text
    assert "2 行" in response.text
    assert "5 列" in response.text
    assert "每月销售额趋势是什么" in response.text
    assert "internal_error" not in response.text


def test_upload_response_includes_recommended_prompts(client: TestClient):
    response = client.post(
        "/upload",
        files={
            "file": (
                "sales_sample.csv",
                (
                    b"order_date,total_amount,product_category,region,channel\n"
                    b"2025-01-01,120.5,Electronics,West,Online\n"
                    b"2025-01-02,88.0,Home,East,Offline\n"
                ),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("recommended_prompts"), list)
    assert payload["recommended_prompts"]


def test_audit_runs_hidden_when_not_enabled(client: TestClient, monkeypatch):
    monkeypatch.setattr(server, "IS_DEVELOPMENT", False)
    monkeypatch.setattr(server, "SETTINGS", replace(server.SETTINGS, audit_api_enabled=False))

    response = client.get("/api/audit/runs")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["code"] == "not_found"


def test_audit_runs_available_in_development(client: TestClient, monkeypatch):
    monkeypatch.setattr(server, "IS_DEVELOPMENT", True)
    monkeypatch.setattr(server, "read_recent_records", lambda limit=100: [{"run_id": "abc"}])

    response = client.get("/api/audit/runs?limit=5")

    assert response.status_code == 200
    assert response.json() == {"runs": [{"run_id": "abc"}], "limit": 5}


def test_dataset_overview_request_writes_chat_route_audit(client: TestClient, monkeypatch, tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=audit_path)
    monkeypatch.setattr(chat_service, "get_audit_logger", lambda: logger)

    upload_response = client.post(
        "/upload",
        files={
            "file": (
                "sales_sample.csv",
                (
                    b"order_date,total_amount,product_category,region,channel\n"
                    b"2025-01-01,120.5,Electronics,West,Online\n"
                    b"2025-01-02,88.0,Home,East,Offline\n"
                ),
                "text/csv",
            )
        },
    )
    assert upload_response.status_code == 200
    dataset_id = upload_response.json()["dataset_id"]

    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "input": {"messages": [{"type": "human", "content": "讲解一下这个数据集"}]},
            "config": {"configurable": {"dataset_id": dataset_id}},
        },
    )

    assert response.status_code == 200
    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    route_record = next(record for record in records if record["tool_name"] == "chat_route")
    assert route_record["tool_args"]["is_dataset_overview"] is True
    assert route_record["extra"]["routing"]["final_intent"] in {"analysis", "followup"}
    assert route_record["extra"]["routing"]["conflict_flags"] is not None
