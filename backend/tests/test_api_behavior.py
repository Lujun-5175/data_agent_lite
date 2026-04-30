from __future__ import annotations

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
    assert set(payload.keys()) == {"error"}
    assert set(payload["error"].keys()) == {"code", "message"}
    assert payload["error"]["code"] == "invalid_file_type"


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
