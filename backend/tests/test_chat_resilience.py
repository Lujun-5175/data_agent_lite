from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.errors import GraphRecursionError

from src import server
from src.agent import _format_dataset_context_summary
from src.data_manager import get_data_context_summary
from src.tools import _serialize_tool_error


def _upload_fixture_dataset(client: TestClient) -> str:
    csv_content = (
        "date,country,state,user_id,sales,orders,churn,contract\n"
        "2024-01-01,US,California,u1,100,1,yes,monthly\n"
        "2024-01-01,US,New York,u2,200,2,no,yearly\n"
        "2024-01-02,US,California,u3,300,3,yes,monthly\n"
        "2024-01-02,US,Texas,u4,50,1,no,monthly\n"
    ).encode("utf-8")
    response = client.post("/upload", files={"file": ("golden.csv", csv_content, "text/csv")})
    assert response.status_code == 200
    return str(response.json()["dataset_id"])


def _parse_sse(response_text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in [item for item in response_text.split("\n\n") if item.strip()]:
        lines = block.splitlines()
        event_line = next((line for line in lines if line.startswith("event: ")), None)
        data_line = next((line for line in lines if line.startswith("data: ")), None)
        assert event_line is not None
        assert data_line is not None
        events.append(
            (
                event_line.replace("event: ", "", 1).strip(),
                json.loads(data_line.replace("data: ", "", 1)),
            )
        )
    return events


class _CaptureGraph:
    def __init__(self) -> None:
        self.last_inputs: dict[str, Any] | None = None

    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        self.last_inputs = inputs
        yield {"event": "on_chain_end", "name": "capture", "data": {}}


class _RetryOnceGraph:
    def __init__(self) -> None:
        self.calls = 0

    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        self.calls += 1
        if self.calls == 1:
            raise httpx.ReadError("boom")
        yield {
            "event": "on_chat_model_stream",
            "name": "retry",
            "data": {"chunk": "恢复后的回答"},
        }


class _LoopingGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        raise GraphRecursionError("loop")
        yield  # pragma: no cover


class _ToolTimeoutGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        yield {"event": "on_tool_start", "name": "python_inter", "data": {}}
        yield {
            "event": "on_chat_model_stream",
            "name": "timeout-tool",
            "data": {
                "chunk": _serialize_tool_error(
                    error_code="tool_execution_timeout",
                    message="代码执行超时（>4.0 秒），请缩小范围。",
                    tool_name="python_inter",
                )
            },
        }
        yield {"event": "on_tool_end", "name": "python_inter", "data": {}}


class _ConnectErrorGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        request = httpx.Request("POST", "https://api.deepseek.com")
        raise httpx.ConnectError("connection refused", request=request)
        yield  # pragma: no cover


class _UnknownModelErrorGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        raise RuntimeError("DeepSeek quota exceeded for current workspace")
        yield  # pragma: no cover


def test_http_error_response_includes_request_id(client: TestClient):
    response = client.post("/chat/stream", json={"input": {"messages": []}})
    assert response.status_code == 422
    payload = response.json()
    assert payload["request_id"]
    assert response.headers["X-Request-ID"] == payload["request_id"]


def test_long_history_is_compressed_before_graph(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    capture_graph = _CaptureGraph()
    monkeypatch.setattr(server, "graph", capture_graph)
    dataset_id = _upload_fixture_dataset(client)
    messages = [
        {"type": "human" if index % 2 == 0 else "assistant", "content": f"message {index} about churn and grouping"}
        for index in range(10)
    ]
    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": messages},
        },
    )
    assert response.status_code == 200
    assert capture_graph.last_inputs is not None
    compressed_messages = capture_graph.last_inputs["messages"]
    assert len(compressed_messages) <= 8
    assert "会话摘要" in compressed_messages[0]["content"]


def test_simple_mode_preserves_summary_and_latest_user_message(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    capture_graph = _CaptureGraph()
    monkeypatch.setattr(server, "graph", capture_graph)
    dataset_id = _upload_fixture_dataset(client)
    messages = [
        {"type": "human", "content": "先分析 churn"},
        {"type": "assistant", "content": "已完成第一步分析"},
        {"type": "human", "content": "现在只看 California"},
        {"type": "assistant", "content": "好的，范围切到 California"},
        {"type": "human", "content": "继续看最近结果为什么这样"},
        {"type": "assistant", "content": "我先复用最近 artifact"},
        {"type": "human", "content": "再总结一下 churn 驱动因素"},
    ]
    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": messages},
        },
    )
    assert response.status_code == 200
    assert capture_graph.last_inputs is not None
    compressed_messages = capture_graph.last_inputs["messages"]
    assert "会话摘要" in compressed_messages[0]["content"]
    assert any(item["type"] == "human" and "再总结一下" in item["content"] for item in compressed_messages)


def test_dataset_context_prompt_is_lightweight(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    prompt = _format_dataset_context_summary(get_data_context_summary(dataset_id))
    assert "Non-Null Count" not in prompt
    assert "数据规模" in prompt
    assert "数值字段" in prompt


def test_stream_retry_once_recovers_and_reports_degradation(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    retry_graph = _RetryOnceGraph()
    monkeypatch.setattr(server, "graph", retry_graph)
    dataset_id = _upload_fixture_dataset(client)
    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": "分析一下 churn"}]},
        },
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert retry_graph.calls == 2
    assert not [payload for event_type, payload in events if event_type == "error"]
    done_payload = next(payload for event_type, payload in events if event_type == "done")
    assert done_payload["degradation_mode"] == "retry_stream_once"
    assert done_payload["request_id"]


def test_stream_error_event_includes_request_id_and_stage(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(server, "graph", _LoopingGraph())
    dataset_id = _upload_fixture_dataset(client)
    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": "做一个非常复杂的多步分析"}]},
        },
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    error_payload = next(payload for event_type, payload in events if event_type == "error")
    assert error_payload["code"] == "agent_recursion_limit"
    assert error_payload["stage"] == "agent_loop"
    assert error_payload["request_id"]
    assert error_payload["retryable"] is False


def test_tool_timeout_becomes_structured_sse_error(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(server, "graph", _ToolTimeoutGraph())
    dataset_id = _upload_fixture_dataset(client)
    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": "分析一下 churn"}]},
        },
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    error_payload = next(payload for event_type, payload in events if event_type == "error")
    assert error_payload["code"] == "tool_execution_timeout"
    assert error_payload["stage"] == "tool_execution"
    assert error_payload["request_id"]


def test_connect_error_is_reported_with_specific_error_code(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(server, "graph", _ConnectErrorGraph())
    dataset_id = _upload_fixture_dataset(client)
    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": "分析一下 churn"}]},
        },
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    error_payload = next(payload for event_type, payload in events if event_type == "error")
    assert error_payload["code"] == "upstream_model_connection_error"
    assert error_payload["stage"] == "model_stream"
    assert "connection refused" in error_payload["message"]


def test_unknown_model_error_preserves_backend_message(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(server, "graph", _UnknownModelErrorGraph())
    dataset_id = _upload_fixture_dataset(client)
    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": "分析一下 churn"}]},
        },
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    error_payload = next(payload for event_type, payload in events if event_type == "error")
    assert error_payload["code"] == "internal_error"
    assert error_payload["stage"] == "model_stream"
    assert error_payload["message"] == "模型调用失败：DeepSeek quota exceeded for current workspace"


@pytest.mark.parametrize("query", ["讲解一下这个数据集", "介绍这份数据", "看看这个表"])
def test_dataset_overview_variants_stream_metadata_without_agent_loop(query: str, client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": query}]},
        },
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[0][0] == "route_info"
    route_info = events[0][1]
    assert route_info["primary_mode"] == "dataset_overview"
    assert route_info["intent_type"] == "dataset_overview"
    assert route_info["final_branch"] == "dataset_overview"
    assert "summarize_dataset" in route_info["requested_capabilities"]
    text = "".join(str(payload.get("content", "")) for event_type, payload in events if event_type == "message_chunk")
    assert "数据规模" in text
    assert "你可以直接点上方推荐问题" in text
    assert not [payload for event_type, payload in events if event_type in {"tool_start", "tool_end"}]
    assert "internal_error" not in response.text


def test_long_history_summary_keeps_older_constraints(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    messages: list[dict[str, str]] = []
    messages.append({"type": "human", "content": "现在只看 California，并且关注 churn 和 sales"})
    messages.append({"type": "assistant", "content": "好的，我会只看 California 并关注 churn/sales"})
    for index in range(16):
        role = "human" if index % 2 == 0 else "assistant"
        messages.append({"type": role, "content": f"filler message {index}"})
    from src.conversation_context import compress_conversation_messages

    compressed = compress_conversation_messages(messages, dataset_id=dataset_id)
    assert compressed.conversation_summary is not None
    summary = compressed.conversation_summary
    assert any("California" in item for item in summary.get("global_constraints", []))
    assert any("churn" in item.lower() or "sales" in item.lower() for item in summary.get("current_goals", []))
