from __future__ import annotations

import json
import os
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_CANARY") != "1",
    reason="Live canary tests are optional. Set RUN_LIVE_CANARY=1 to enable.",
)


def _upload_fixture_dataset(client: TestClient) -> str:
    csv_content = (
        "date,country,state,user_id,sales,orders\n"
        "2024-01-01,US,California,u1,100,1\n"
        "2024-01-01,US,New York,u2,200,2\n"
        "2024-01-02,US,California,u3,300,3\n"
        "2024-01-02,US,Texas,u4,50,1\n"
        "2024-01-08,US,California,u5,120,1\n"
        "2024-02-01,US,New York,u6,80,1\n"
        "2024-02-02,CA,Ontario,u7,500,5\n"
        "2024-02-03,US,California,u8,400,4\n"
    ).encode("utf-8")
    response = client.post("/upload", files={"file": ("canary.csv", csv_content, "text/csv")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"]
    return str(payload["dataset_id"])


def _stream_events(client: TestClient, dataset_id: str | None, messages: list[dict[str, str]]):
    body: dict[str, Any] = {"input": {"messages": messages}}
    if dataset_id:
        body["dataset_id"] = dataset_id
        body["config"] = {"configurable": {"dataset_id": dataset_id}}

    response = client.post("/chat/stream", json=body)
    if response.status_code != 200:
        return response, []

    blocks = [block for block in response.text.split("\n\n") if block.strip()]
    events: list[tuple[str, dict[str, Any]]] = []
    for block in blocks:
        lines = block.splitlines()
        event_line = next((line for line in lines if line.startswith("event: ")), None)
        data_line = next((line for line in lines if line.startswith("data: ")), None)
        if not event_line or not data_line:
            continue
        event_type = event_line.replace("event: ", "", 1).strip()
        payload = json.loads(data_line.replace("data: ", "", 1))
        events.append((event_type, payload))
    return response, events


def _joined_text(events: list[tuple[str, dict[str, Any]]]) -> str:
    parts: list[str] = []
    for event_type, payload in events:
        if event_type == "message_chunk":
            content = payload.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "".join(parts)


def _normalize_number_text(text: str) -> str:
    lowered = text.lower()
    return re.sub(r"[\s,，]", "", lowered)


@pytest.fixture(autouse=True)
def _require_live_key():
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for live canary tests.")


def test_live_canary_basic_metric_query(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    _, events = _stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "请计算总 sales、总行数和平均 sales，并简要说明。"}],
    )
    text = _normalize_number_text(_joined_text(events))
    assert text
    assert ("1750" in text) or ("总sales为1750" in text)
    assert ("8" in text) or ("总行数8" in text)
    assert ("218.75" in text) or ("平均" in text and "sales" in text)


def test_live_canary_filter_aggregation(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    _, events = _stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "只看 US 用户，按 state 分组计算平均 sales，按均值降序，返回前 10。"}],
    )
    text = _normalize_number_text(_joined_text(events))
    assert text
    assert "california" in text
    assert "newyork" in text or "newyork" in text.replace(" ", "")
    assert "texas" in text
    assert ("230" in text) or ("230.0" in text)


def test_live_canary_chart_generation(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    _, events = _stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "请生成一个 US 用户按州的 sales 柱状图，并简要说明。"}],
    )
    image_events = [payload for event_type, payload in events if event_type == "image_generated"]
    assert image_events, "expected at least one image_generated event"
    image_payload = image_events[-1]
    assert isinstance(image_payload.get("filename"), str)
    assert isinstance(image_payload.get("image_url"), str)
    assert "/static/images/" in str(image_payload.get("image_url"))


def test_live_canary_follow_up_query(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    _, events = _stream_events(
        client,
        dataset_id,
        [
            {"type": "human", "content": "先算 US 用户的总 sales。"},
            {"type": "assistant", "content": "好的，我来计算。"},
            {"type": "human", "content": "现在只看 California。"},
        ],
    )
    text = _normalize_number_text(_joined_text(events))
    assert "california" in text
    assert ("920" in text) or ("总sales" in text and "california" in text)


def test_live_canary_error_path_structured(client: TestClient):
    response, _ = _stream_events(
        client,
        None,
        [{"type": "human", "content": "请根据当前数据集做相关性分析。"}],
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "dataset_required"
    assert "上传 CSV" in payload["error"]["message"]
