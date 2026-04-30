from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src import intent_planner
from src import server
from src.data_manager import get_dataframe
from src.result_types import artifact_registry, build_artifact
from src.tools import StatsHelperAPI
from src.tools import set_current_image_event


def _upload_telco_fixture_dataset(client: TestClient) -> str:
    csv_content = (
        "customerID,tenure,MonthlyCharges,TotalCharges,Churn,Contract,gender\n"
        "c1,1,29.85,29.85,No,Month-to-month,Female\n"
        "c2,34,56.95,1889.5,No,One year,Male\n"
        "c3,2,53.85,108.15,Yes,Month-to-month,Female\n"
        "c4,45,42.3,1840.75,No,Two year,Male\n"
        "c5,8,70.7,568.35,Yes,Month-to-month,Female\n"
        "c6,22,89.1,1949.4,No,One year,Male\n"
        "c7,10,29.75,301.9,Yes,Month-to-month,Female\n"
        "c8,60,109.9,6660.2,No,Two year,Male\n"
        "c9,5,79.35,401.45,Yes,Month-to-month,Female\n"
        "c10,72,99.65,7251.7,No,Two year,Male\n"
    ).encode("utf-8")
    response = client.post("/upload", files={"file": ("telco.csv", csv_content, "text/csv")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"]
    return str(payload["dataset_id"])


def _stream_events(client: TestClient, dataset_id: str, messages: list[dict[str, str]]) -> list[tuple[str, dict[str, Any]]]:
    set_current_image_event(None)
    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": messages},
        },
    )
    assert response.status_code == 200

    events: list[tuple[str, dict[str, Any]]] = []
    for block in [item for item in response.text.split("\n\n") if item.strip()]:
        lines = block.splitlines()
        event_line = next((line for line in lines if line.startswith("event: ")), None)
        data_line = next((line for line in lines if line.startswith("data: ")), None)
        assert event_line is not None
        assert data_line is not None
        event_type = event_line.replace("event: ", "", 1).strip()
        payload = json.loads(data_line.replace("data: ", "", 1))
        events.append((event_type, payload))
    return events


class _SilentGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        yield {
            "event": "on_chat_model_stream",
            "name": "golden-model",
            "data": {
                "chunk": "让我先查看一下更简单的方式，然后再尝试。",
            },
        }
        yield {"event": "on_chain_end", "name": "golden-chain", "data": {}}


class _NoImageChartGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        yield {"event": "on_tool_start", "name": "fig_inter", "data": {}}
        yield {
            "event": "on_chat_model_stream",
            "name": "golden-model",
            "data": {"chunk": "图表已经准备好了。"},
        }
        yield {"event": "on_tool_end", "name": "fig_inter", "data": {}}


class _AnalysisOnlyGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        yield {"event": "on_tool_start", "name": "python_inter", "data": {}}
        yield {
            "event": "on_chat_model_stream",
            "name": "golden-model",
            "data": {"chunk": "已完成分组比较和探索性分析。"},
        }
        yield {"event": "on_tool_end", "name": "python_inter", "data": {}}


class _InternalPlannerLeakGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        intent_payload = json.dumps(
            {
                "intent_type": "followup",
                "requires_ml": False,
                "requires_chart": False,
                "requires_python_analysis": False,
                "deliverables": [],
                "reasoning_summary": "internal planner note",
                "suggested_plan": ["do not show this"],
            },
            ensure_ascii=False,
        )
        yield {"event": "on_chain_stream", "name": "intent-planner", "data": {"chunk": intent_payload}}
        yield {"event": "on_chat_model_stream", "name": "intent-planner", "data": {"chunk": intent_payload}}
        yield {
            "event": "on_chat_model_stream",
            "name": "golden-model",
            "data": {"chunk": "以下是该数据集的完整讲解。"},
        }
        yield {"event": "on_chain_end", "name": "golden-chain", "data": {}}


class _InternalPlannerPrefixLeakGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        intent_payload = json.dumps(
            {
                "intent_type": "followup",
                "requires_ml": False,
                "requires_chart": False,
                "requires_python_analysis": False,
                "deliverables": ["summary"],
                "reasoning_summary": "internal planner note",
                "suggested_plan": "do not show this",
            },
            ensure_ascii=False,
            indent=2,
        )
        yield {
            "event": "on_chat_model_stream",
            "name": "golden-model",
            "data": {"chunk": f"{intent_payload}好的，我来对这个数据集进行全面讲解。"},
        }


class _MixedWorkflowGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        dataset_id = config.get("configurable", {}).get("dataset_id")
        yield {"event": "on_tool_start", "name": "python_inter", "data": {}}
        yield {
            "event": "on_chat_model_stream",
            "name": "golden-model",
            "data": {"chunk": "先做探索性分析。"},
        }
        yield {"event": "on_tool_end", "name": "python_inter", "data": {}}
        yield {"event": "on_tool_start", "name": "ml_execute", "data": {"action": "train"}}
        if isinstance(dataset_id, str) and dataset_id:
            artifact_registry.register(
                dataset_id,
                build_artifact(
                    artifact_type="model_result",
                    dataset_id=dataset_id,
                    payload={"target": "Churn", "model_type": "logistic_regression", "metrics": {"accuracy": 0.8}},
                ),
            )
        yield {
            "event": "on_chat_model_stream",
            "name": "golden-model",
            "data": {"chunk": json.dumps({"artifact_type": "model_result", "artifact_id": "demo", "metrics": {"accuracy": 0.8}}, ensure_ascii=False)},
        }
        yield {"event": "on_tool_end", "name": "ml_execute", "data": {"action": "train"}}


class _CaptureGraph:
    def __init__(self) -> None:
        self.last_inputs: dict[str, Any] | None = None

    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        self.last_inputs = inputs
        yield {"event": "on_chain_end", "name": "golden-chain", "data": {}}


class _PlannerModelThatOvercallsML:
    def invoke(self, messages, **kwargs):
        return type(
            "Response",
            (),
            {
                "content": json.dumps(
                    {
                        "intent_type": "ml",
                        "requires_ml": True,
                        "requires_chart": False,
                        "requires_python_analysis": False,
                        "deliverables": ["metrics"],
                        "reasoning_summary": "强行判定为建模。",
                        "suggested_plan": ["train a model", "report metrics"],
                    },
                    ensure_ascii=False,
                )
            },
        )()


@pytest.mark.usefixtures("client")
def test_ml_stream_requires_structured_artifact_and_suppresses_fallback(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(server, "graph", _SilentGraph())
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "做一个逻辑回归预测 churn，给我 accuracy 和重要特征"}],
    )

    joined_text = "".join(
        str(payload.get("content", ""))
        for event_type, payload in events
        if event_type == "message_chunk"
    )
    assert "让我先查看一下" not in joined_text

    errors = [payload for event_type, payload in events if event_type == "error"]
    assert errors
    assert errors[-1]["code"] == "structured_failure"
    assert "ml_execute" in errors[-1]["message"] or "结构化结果" in errors[-1]["message"]


@pytest.mark.usefixtures("client")
def test_chart_stream_requires_image_artifact(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(server, "graph", _NoImageChartGraph())
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "请生成按 gender 的 churn rate 柱状图，并简要说明"}],
    )

    image_events = [payload for event_type, payload in events if event_type == "image_generated"]
    assert not image_events
    errors = [payload for event_type, payload in events if event_type == "error"]
    assert errors
    assert errors[-1]["code"] == "structured_failure"
    assert "图片结果" in errors[-1]["message"]


@pytest.mark.usefixtures("client")
def test_exploratory_analysis_request_is_not_forced_into_ml_validation(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(server, "graph", _AnalysisOnlyGraph())
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", _PlannerModelThatOvercallsML())
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "analyze churn drivers"}],
    )

    errors = [payload for event_type, payload in events if event_type == "error"]
    assert not errors
    text = "".join(str(payload.get("content", "")) for event_type, payload in events if event_type == "message_chunk")
    assert "探索性分析" in text or "分组比较" in text


@pytest.mark.usefixtures("client")
def test_stream_suppresses_internal_intent_payloads(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(server, "graph", _InternalPlannerLeakGraph())
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "讲解一下这个数据集"}],
    )

    text = "".join(str(payload.get("content", "")) for event_type, payload in events if event_type == "message_chunk")
    assert "以下是该数据集的完整讲解" in text
    assert "intent_type" not in text
    assert "reasoning_summary" not in text
    assert "suggested_plan" not in text


@pytest.mark.usefixtures("client")
def test_stream_strips_internal_intent_payload_prefix(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(server, "graph", _InternalPlannerPrefixLeakGraph())
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "请全面分析这个数据集"}],
    )

    text = "".join(str(payload.get("content", "")) for event_type, payload in events if event_type == "message_chunk")
    assert text == "好的，我来对这个数据集进行全面讲解。"
    assert "intent_type" not in text
    assert "reasoning_summary" not in text


@pytest.mark.usefixtures("client")
def test_mixed_workflow_can_chain_analysis_then_ml(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(server, "graph", _MixedWorkflowGraph())
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "analyze churn drivers and, if useful, try a simple model"}],
    )

    tool_names = [payload.get("tool_name") for event_type, payload in events if event_type in {"tool_start", "tool_end"}]
    assert "python_inter" in tool_names
    assert "ml_execute" in tool_names
    errors = [payload for event_type, payload in events if event_type == "error"]
    assert not errors or errors[-1]["code"] != "structured_failure"


@pytest.mark.usefixtures("client")
def test_follow_up_prompt_receives_recent_result_context(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    dataset_id = _upload_telco_fixture_dataset(client)
    df = get_dataframe(dataset_id)
    stats = StatsHelperAPI(df, dataset_id=dataset_id)
    stats.group_summary(
        group_by="Contract",
        metrics=[{"op": "rate", "column": "Churn", "as": "churn_rate"}],
        sort_by="churn_rate",
        ascending=False,
        top_n=3,
    )

    capture_graph = _CaptureGraph()
    monkeypatch.setattr(server, "graph", capture_graph)
    _stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "解释一下刚才的结果"}],
    )

    assert capture_graph.last_inputs is not None
    messages = capture_graph.last_inputs.get("messages", [])
    assistant_messages = [
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("type") == "assistant"
    ]
    assert any("最近一次结构化结果" in content for content in assistant_messages)
