from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src import server
from src.data_manager import get_dataframe
from src.errors import AppError
from src.routing_rules import RoutingContext, decide_ml_intent
from src.tools import MLHelperAPI


ML_GOLDEN_PROMPTS: list[tuple[str, str]] = [
    ("case_1", "做一下逻辑回归预测一下流失率"),
    ("case_2", "用逻辑回归预测客户流失，看看哪些变量最重要"),
    ("case_3", "基于这份数据训练一个 baseline logistic regression 模型预测 Churn"),
    ("case_4", "做个 churn prediction，给我 accuracy 和重要特征"),
    ("case_5", "先做逻辑回归，再告诉我模型指标"),
]


FORBIDDEN_FALLBACK_PHRASES = [
    "让我先查看一下",
    "让我用更安全的方式",
    "让我尝试更简单的方法",
    "由于当前环境限制，我无法直接执行完整建模流程",
    "数据适合建模",
    "需要预处理",
    "环境限制无法执行",
]


def _upload_telco_fixture_dataset(client: TestClient) -> str:
    csv_content = (
        "customerID,tenure,MonthlyCharges,TotalCharges,Churn,Contract\n"
        "c1,1,29.85,29.85,No,Month-to-month\n"
        "c2,34,56.95,1889.5,No,One year\n"
        "c3,2,53.85,108.15,Yes,Month-to-month\n"
        "c4,45,42.3,1840.75,No,Two year\n"
        "c5,8,70.7,568.35,Yes,Month-to-month\n"
        "c6,22,89.1,1949.4,No,One year\n"
        "c7,10,29.75,301.9,Yes,Month-to-month\n"
        "c8,60,109.9,6660.2,No,Two year\n"
        "c9,5,79.35,401.45,Yes,Month-to-month\n"
        "c10,72,99.65,7251.7,No,Two year\n"
    ).encode("utf-8")
    response = client.post("/upload", files={"file": ("telco.csv", csv_content, "text/csv")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"]
    return str(payload["dataset_id"])


def _extract_messages(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    messages = inputs.get("messages", [])
    return [item for item in messages if isinstance(item, dict)]


def _latest_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("type") in {"human", "user"}:
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _chat_stream_events(client: TestClient, dataset_id: str, messages: list[dict[str, str]]) -> list[tuple[str, dict[str, Any]]]:
    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": messages},
        },
    )
    assert response.status_code == 200
    blocks = [block for block in response.text.split("\n\n") if block.strip()]
    events: list[tuple[str, dict[str, Any]]] = []
    for block in blocks:
        lines = block.splitlines()
        event_line = next((line for line in lines if line.startswith("event: ")), None)
        data_line = next((line for line in lines if line.startswith("data: ")), None)
        assert event_line is not None
        assert data_line is not None
        event_type = event_line.replace("event: ", "", 1).strip()
        payload = json.loads(data_line.replace("data: ", "", 1))
        events.append((event_type, payload))
    return events


def _joined_message_chunks(events: list[tuple[str, dict[str, Any]]]) -> str:
    return "".join(
        str(payload.get("content", ""))
        for event_type, payload in events
        if event_type == "message_chunk"
    )


def _latest_chunk_json(events: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    chunks = [payload for event_type, payload in events if event_type == "message_chunk"]
    assert chunks, "expected at least one message_chunk event"
    content = chunks[-1]["content"]
    assert isinstance(content, str)
    return json.loads(content)


class NaturalLanguageMLWorkflowGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        dataset_id = config.get("configurable", {}).get("dataset_id")
        if not dataset_id:
            raise AppError("dataset_required", "当前未选择数据集，请先上传 CSV 文件后再进行数据分析。", 400)

        df = get_dataframe(str(dataset_id))
        ml = MLHelperAPI(df, dataset_id=str(dataset_id))
        messages = _extract_messages(inputs)
        latest = _latest_user_message(messages).strip()
        latest_lower = latest.lower()
        decision = decide_ml_intent(RoutingContext(message=latest))

        if not decision.matched:
            yield {
                "event": "on_chat_model_stream",
                "name": "golden-model",
                "data": {"chunk": json.dumps({"artifact_type": "structured_failure", "reason": "ml_intent_not_matched"}, ensure_ascii=False)},
            }
            yield {"event": "on_tool_end", "name": "ml.route_guard", "data": {}}
            return

        yield {"event": "on_tool_start", "name": "ml_execute", "data": {"action": "train"}}
        try:
            model_artifact = ml.logistic_fit(target="Churn", positive_label="Yes")
        except Exception as exc:
            yield {
                "event": "on_chat_model_stream",
                "name": "golden-model",
                "data": {
                    "chunk": json.dumps(
                        {
                            "artifact_type": "structured_failure",
                            "stage": "model_fit",
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                },
            }
            yield {"event": "on_tool_end", "name": "ml_execute", "data": {"action": "train"}}
            return

        yield {
            "event": "on_chat_model_stream",
            "name": "golden-model",
            "data": {"chunk": json.dumps(model_artifact, ensure_ascii=False)},
        }
        yield {"event": "on_tool_end", "name": "ml_execute", "data": {"action": "train"}}

        if any(token in latest_lower for token in ("accuracy", "指标", "model metrics", "模型指标", "metrics")):
            yield {"event": "on_tool_start", "name": "ml_execute", "data": {"action": "metrics"}}
            metrics_artifact = ml.metrics(model_artifact_id=model_artifact["artifact_id"])
            yield {
                "event": "on_chat_model_stream",
                "name": "golden-model",
                "data": {"chunk": json.dumps(metrics_artifact, ensure_ascii=False)},
            }
            yield {"event": "on_tool_end", "name": "ml_execute", "data": {"action": "metrics"}}

        if any(token in latest for token in ("重要特征", "哪些变量", "feature importance", "variables")):
            yield {"event": "on_tool_start", "name": "ml_execute", "data": {"action": "feature_importance"}}
            fi_artifact = ml.feature_importance(model_artifact_id=model_artifact["artifact_id"], top_k=5)
            yield {
                "event": "on_chat_model_stream",
                "name": "golden-model",
                "data": {"chunk": json.dumps(fi_artifact, ensure_ascii=False)},
            }
            yield {"event": "on_tool_end", "name": "ml_execute", "data": {"action": "feature_importance"}}


@pytest.fixture
def natural_language_ml_graph(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "graph", NaturalLanguageMLWorkflowGraph())


@pytest.mark.parametrize("case_id,prompt", ML_GOLDEN_PROMPTS)
def test_ml_intent_routes_all_natural_language_cases(case_id: str, prompt: str):
    decision = decide_ml_intent(RoutingContext(message=prompt))
    assert decision.matched is True, case_id
    assert decision.score >= decision.threshold, case_id
    assert any("weighted keyword" in reason for reason in decision.reasons), case_id


@pytest.mark.usefixtures("natural_language_ml_graph")
@pytest.mark.parametrize(
    "prompt,expected_artifact_types,expected_tool_names",
    [
        ("做一下逻辑回归预测一下流失率", {"model_result"}, {"ml_execute"}),
        ("用逻辑回归预测客户流失，看看哪些变量最重要", {"model_result", "feature_importance_result"}, {"ml_execute"}),
        ("基于这份数据训练一个 baseline logistic regression 模型预测 Churn", {"model_result"}, {"ml_execute"}),
        ("做个 churn prediction，给我 accuracy 和重要特征", {"model_result", "metrics_result", "feature_importance_result"}, {"ml_execute"}),
        ("先做逻辑回归，再告诉我模型指标", {"model_result", "metrics_result"}, {"ml_execute"}),
    ],
)
def test_natural_language_ml_baseline_runs_structured_ml_path(
    client: TestClient,
    prompt: str,
    expected_artifact_types: set[str],
    expected_tool_names: set[str],
):
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _chat_stream_events(client, dataset_id, [{"type": "human", "content": prompt}])

    event_names: set[str | None] = set()
    for event_type, payload in events:
        if event_type in {"tool_start", "tool_end"}:
            event_names.add(payload.get("tool_name") or payload.get("name"))
    assert expected_tool_names.issubset(event_names)

    text = _joined_message_chunks(events)
    for phrase in FORBIDDEN_FALLBACK_PHRASES:
        assert phrase not in text

    artifact_types: set[str | None] = set()
    for event_type, payload in events:
        if event_type != "message_chunk":
            continue
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip().startswith("{"):
            continue
        artifact = json.loads(content)
        artifact_types.add(artifact.get("artifact_type"))
    assert expected_artifact_types.intersection(artifact_types)

    if "metrics_result" in expected_artifact_types:
        metrics_artifact = None
        for event_type, payload in events:
            if event_type != "message_chunk":
                continue
            content = payload.get("content")
            if not isinstance(content, str) or not content.strip().startswith("{"):
                continue
            artifact = json.loads(content)
            if artifact.get("artifact_type") == "metrics_result":
                metrics_artifact = artifact
                break
        assert metrics_artifact is not None
        assert "accuracy" in metrics_artifact["metrics"]

    if "feature_importance_result" in expected_artifact_types:
        fi_artifact = None
        for event_type, payload in events:
            if event_type != "message_chunk":
                continue
            content = payload.get("content")
            if not isinstance(content, str) or not content.strip().startswith("{"):
                continue
            artifact = json.loads(content)
            if artifact.get("artifact_type") == "feature_importance_result":
                fi_artifact = artifact
                break
        assert fi_artifact is not None
        assert fi_artifact["items"]


@pytest.mark.usefixtures("natural_language_ml_graph")
def test_natural_language_ml_output_never_leaks_fallback_language(client: TestClient):
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "做个 churn prediction，给我 accuracy 和重要特征"}],
    )
    text = _joined_message_chunks(events)
    for phrase in FORBIDDEN_FALLBACK_PHRASES:
        assert phrase not in text


@pytest.mark.usefixtures("natural_language_ml_graph")
def test_natural_language_ml_helper_failure_returns_structured_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    def _boom(*args: Any, **kwargs: Any):
        raise RuntimeError("model backend down")

    monkeypatch.setattr(MLHelperAPI, "logistic_fit", _boom)

    dataset_id = _upload_telco_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "做一下逻辑回归预测一下流失率"}],
    )
    errors = [payload for event_type, payload in events if event_type == "error"]
    assert errors
    assert errors[-1]["code"] == "structured_failure"
    assert "model_result" in errors[-1]["message"] or "ml_execute" in errors[-1]["message"]
    text = _joined_message_chunks(events)
    for phrase in FORBIDDEN_FALLBACK_PHRASES:
        assert phrase not in text
