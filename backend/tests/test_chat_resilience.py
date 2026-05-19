from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.errors import GraphRecursionError

from src import chat_service, intent_planner, server
from src.agent import _format_dataset_context_summary
from src.data_manager import get_data_context_summary
from src.routing_models import RoutingDecision
from src.task_plan_models import TaskPlan, TaskSpec
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


def _upload_plan_execution_dataset(client: TestClient) -> str:
    csv_content = (
        "channel_source,session_count,page_views,conversion_flag,ab_group\n"
        "organic,2,5,1,A\n"
        "organic,4,8,0,B\n"
        "paid,6,10,1,A\n"
        "paid,8,12,1,B\n"
    ).encode("utf-8")
    response = client.post("/upload", files={"file": ("plan.csv", csv_content, "text/csv")})
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
        self.last_context: Any | None = None
        self.calls = 0

    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        self.calls += 1
        self.last_inputs = inputs
        self.last_context = context
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


def test_graph_receives_precomputed_routing_decision(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    capture_graph = _CaptureGraph()
    monkeypatch.setattr(server, "graph", capture_graph)
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
    assert capture_graph.last_context is not None
    assert capture_graph.last_context.dataset_id == dataset_id
    assert isinstance(capture_graph.last_context.routing_decision, dict)
    assert "primary_mode" in capture_graph.last_context.routing_decision


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


def test_route_info_includes_task_plan_metadata(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    monkeypatch.setattr(
        chat_service,
        "plan_request_with_llm",
        lambda *args, **kwargs: intent_planner.UnifiedPlanningResult(
            routing_decision=RoutingDecision(
                primary_mode="analysis",
                confidence_score=0.84,
                confidence_band="high",
                needs_dataset=True,
                needs_tool_execution=True,
                needs_artifact_context=False,
                requested_capabilities=["group_analysis", "stat_test", "python_analysis"],
                ambiguity_flags=[],
                guardrail_actions=[],
                fallback_reasons=[],
                reasoning_summary="用户需要分析 A/B 组差异。",
                execution_plan=["group rows", "compare rates"],
                deliverables=["summary"],
                route_source="llm_primary",
            ),
            task_plan=TaskPlan(
                goal="比较 A/B 组转化差异并补充均值统计",
                planning_confidence=0.83,
                assumptions=["conversion_flag 为二元转化标记"],
                ambiguity_flags=[],
                tasks=[
                    TaskSpec(
                        task_id="task_1",
                        task_type="group_aggregate",
                        description="比较 ab_group 的 conversion_flag 转化率",
                        inputs={"group_by": ["ab_group"]},
                        required_outputs=["conversion_rate_comparison"],
                    ),
                    TaskSpec(
                        task_id="task_2",
                        task_type="python_analysis",
                        description="执行卡方检验评估差异显著性",
                        inputs={"test": "chi_square"},
                        depends_on=["task_1"],
                        required_outputs=["chi_square_result"],
                    ),
                ],
                final_response_style="concise_analysis",
            ),
        ),
    )

    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": "分析 ab_group 中 A/B 的 conversion_flag 差异"}]},
        },
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    route_info = next(payload for event_type, payload in events if event_type == "route_info")
    assert route_info["task_plan_available"] is True
    assert route_info["task_plan_goal"] == "比较 A/B 组转化差异并补充均值统计"
    assert route_info["task_plan_confidence"] == pytest.approx(0.83)
    assert route_info["task_plan_tasks"][0]["task_type"] == "group_aggregate"
    assert route_info["task_plan_tasks"][1]["depends_on"] == ["task_1"]


def test_analyze_chat_request_attaches_task_plan(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    monkeypatch.setattr(
        chat_service,
        "plan_request_with_llm",
        lambda *args, **kwargs: intent_planner.UnifiedPlanningResult(
            routing_decision=RoutingDecision(
                primary_mode="analysis",
                confidence_score=0.81,
                confidence_band="high",
                needs_dataset=True,
                needs_tool_execution=True,
                needs_artifact_context=False,
                requested_capabilities=["group_analysis", "python_analysis"],
                ambiguity_flags=[],
                guardrail_actions=[],
                fallback_reasons=[],
                reasoning_summary="用户需要按渠道分组分析。",
                execution_plan=["group rows", "summarize metrics"],
                deliverables=["summary"],
                route_source="llm_primary",
            ),
            task_plan=TaskPlan(
                goal="按 channel_source 分组统计均值和转化率",
                planning_confidence=0.79,
                assumptions=[],
                ambiguity_flags=[],
                tasks=[
                    TaskSpec(
                        task_id="task_1",
                        task_type="group_aggregate",
                        description="按渠道统计 session_count/page_views 均值和 conversion_flag 转化率",
                        inputs={"group_by": ["channel_source"]},
                        required_outputs=["group_summary_table"],
                    )
                ],
                final_response_style="concise_analysis",
            ),
        ),
    )

    requirements = chat_service.analyze_chat_request(
        {
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {
                "messages": [
                    {
                        "type": "human",
                        "content": "按 channel_source 分组，统计 session_count、page_views 均值，并比较 conversion_flag 转化率",
                    }
                ]
            },
        }
    )

    assert requirements.task_plan is not None
    assert requirements.task_plan.goal == "按 channel_source 分组统计均值和转化率"
    assert requirements.task_plan.tasks[0].task_type == "group_aggregate"


def test_supported_task_plan_executes_before_graph(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    capture_graph = _CaptureGraph()
    monkeypatch.setattr(server, "graph", capture_graph)
    dataset_id = _upload_plan_execution_dataset(client)
    monkeypatch.setattr(
        chat_service,
        "plan_request_with_llm",
        lambda *args, **kwargs: intent_planner.UnifiedPlanningResult(
            routing_decision=RoutingDecision(
                primary_mode="analysis",
                confidence_score=0.88,
                confidence_band="high",
                needs_dataset=True,
                needs_tool_execution=True,
                needs_artifact_context=False,
                requested_capabilities=["group_analysis", "python_analysis"],
                ambiguity_flags=[],
                guardrail_actions=[],
                fallback_reasons=[],
                reasoning_summary="用户需要按渠道分组统计。",
                execution_plan=["group rows", "summarize metrics"],
                deliverables=["summary"],
                route_source="llm_primary",
            ),
            task_plan=TaskPlan(
                goal="按 channel_source 分组统计均值和转化率",
                planning_confidence=0.84,
                assumptions=["conversion_flag 为 0/1 二元列"],
                ambiguity_flags=[],
                tasks=[
                    TaskSpec(
                        task_id="task_1",
                        task_type="group_aggregate",
                        description="按渠道统计 session_count/page_views 均值和 conversion_flag 转化率",
                        inputs={
                            "group_by": ["channel_source"],
                            "metrics": [
                                {"column": "session_count", "agg": "mean"},
                                {"column": "page_views", "agg": "mean"},
                                {"column": "conversion_flag", "agg": "mean", "semantic": "rate"},
                            ],
                        },
                        required_outputs=["group_summary_table"],
                    )
                ],
                final_response_style="concise_analysis",
            ),
        ),
    )

    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": "按 channel_source 分组分析 session_count 和转化率"}]},
        },
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    route_info = next(payload for event_type, payload in events if event_type == "route_info")
    assert route_info["final_branch"] == "plan_execution"
    text = "".join(str(payload.get("content", "")) for event_type, payload in events if event_type == "message_chunk")
    assert "已按统一计划执行" in text
    assert "channel_source" in text
    assert "conversion_flag_rate" in text
    assert capture_graph.calls == 0


def test_unsupported_task_plan_falls_back_to_graph(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    capture_graph = _CaptureGraph()
    monkeypatch.setattr(server, "graph", capture_graph)
    dataset_id = _upload_plan_execution_dataset(client)
    monkeypatch.setattr(
        chat_service,
        "plan_request_with_llm",
        lambda *args, **kwargs: intent_planner.UnifiedPlanningResult(
            routing_decision=RoutingDecision(
                primary_mode="analysis",
                confidence_score=0.79,
                confidence_band="high",
                needs_dataset=True,
                needs_tool_execution=True,
                needs_artifact_context=False,
                requested_capabilities=["group_analysis", "stat_test"],
                ambiguity_flags=[],
                guardrail_actions=[],
                fallback_reasons=[],
                reasoning_summary="用户需要比较分组差异。",
                execution_plan=["compare groups", "run test"],
                deliverables=["summary"],
                route_source="llm_primary",
            ),
            task_plan=TaskPlan(
                goal="比较 A/B 组转化差异",
                planning_confidence=0.8,
                assumptions=[],
                ambiguity_flags=[],
                tasks=[
                    TaskSpec(
                        task_id="task_1",
                        task_type="python_analysis",
                        description="执行卡方检验",
                        inputs={"test": "chi_square"},
                        required_outputs=["chi_square_result"],
                    )
                ],
                final_response_style="concise_analysis",
            ),
        ),
    )

    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": "比较 A/B 组差异"}]},
        },
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    route_info = next(payload for event_type, payload in events if event_type == "route_info")
    assert route_info["final_branch"] == "agent_graph"
    assert capture_graph.calls == 1


def test_mixed_task_plan_executes_supported_subset_before_graph(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    capture_graph = _CaptureGraph()
    monkeypatch.setattr(server, "graph", capture_graph)
    dataset_id = _upload_plan_execution_dataset(client)
    monkeypatch.setattr(
        chat_service,
        "plan_request_with_llm",
        lambda *args, **kwargs: intent_planner.UnifiedPlanningResult(
            routing_decision=RoutingDecision(
                primary_mode="analysis",
                confidence_score=0.72,
                confidence_band="medium",
                needs_dataset=True,
                needs_tool_execution=True,
                needs_artifact_context=False,
                requested_capabilities=["group_analysis", "python_analysis"],
                ambiguity_flags=["clarification_requested"],
                guardrail_actions=["fallback_to_heuristic_due_to_low_confidence"],
                fallback_reasons=["low_confidence_with_ambiguity"],
                reasoning_summary="先做分组统计，再考虑补充检验。",
                execution_plan=["group rows", "run significance test"],
                deliverables=["summary"],
                route_source="llm_with_guardrail",
            ),
            task_plan=TaskPlan(
                goal="按渠道分组统计并补充显著性检验",
                planning_confidence=0.71,
                assumptions=["conversion_flag 为 0/1 二元列"],
                ambiguity_flags=[],
                tasks=[
                    TaskSpec(
                        task_id="task_1",
                        task_type="group_aggregate",
                        description="按渠道统计 session_count/page_views 均值和 conversion_flag 转化率",
                        inputs={
                            "group_by": ["channel_source"],
                            "metrics": [
                                {"column": "session_count", "agg": "mean"},
                                {"column": "page_views", "agg": "mean"},
                                {"column": "conversion_flag", "agg": "mean", "semantic": "rate"},
                            ],
                        },
                        required_outputs=["group_summary_table"],
                    ),
                    TaskSpec(
                        task_id="task_2",
                        task_type="python_analysis",
                        description="对转化差异做进一步显著性检验",
                        inputs={"test": "chi_square"},
                        depends_on=["task_1"],
                        required_outputs=["chi_square_result"],
                    ),
                ],
                final_response_style="concise_analysis",
            ),
        ),
    )

    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": "按 channel_source 分组分析 session_count 和转化率，并判断是否显著"}]},
        },
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    route_info = next(payload for event_type, payload in events if event_type == "route_info")
    assert route_info["final_branch"] == "plan_execution"
    text = "".join(str(payload.get("content", "")) for event_type, payload in events if event_type == "message_chunk")
    assert "已按统一计划执行" in text
    assert "当前先执行了 1 个可直接落地的计划任务" in text
    assert "conversion_flag_rate" in text
    assert capture_graph.calls == 0


def test_plan_verifier_reports_incomplete_required_outputs(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    capture_graph = _CaptureGraph()
    monkeypatch.setattr(server, "graph", capture_graph)
    dataset_id = _upload_plan_execution_dataset(client)
    monkeypatch.setattr(
        chat_service,
        "plan_request_with_llm",
        lambda *args, **kwargs: intent_planner.UnifiedPlanningResult(
            routing_decision=RoutingDecision(
                primary_mode="analysis",
                confidence_score=0.82,
                confidence_band="high",
                needs_dataset=True,
                needs_tool_execution=True,
                needs_artifact_context=False,
                requested_capabilities=["group_analysis"],
                ambiguity_flags=[],
                guardrail_actions=[],
                fallback_reasons=[],
                reasoning_summary="用户需要输出分组结果表。",
                execution_plan=["summarize dataset"],
                deliverables=["summary"],
                route_source="llm_primary",
            ),
            task_plan=TaskPlan(
                goal="输出数据集概况并返回分组结果表",
                planning_confidence=0.74,
                assumptions=[],
                ambiguity_flags=[],
                tasks=[
                    TaskSpec(
                        task_id="task_1",
                        task_type="dataset_summary",
                        description="输出数据集概况",
                        inputs={},
                        required_outputs=["group_summary_table"],
                    )
                ],
                final_response_style="concise_analysis",
            ),
        ),
    )

    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": "给我一个分组结果表"}]},
        },
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    route_info = next(payload for event_type, payload in events if event_type == "route_info")
    assert route_info["final_branch"] == "plan_execution"
    error_payload = next(payload for event_type, payload in events if event_type == "error")
    assert error_payload["code"] == "task_plan_incomplete"
    assert error_payload["stage"] == "plan_verification"
    assert "group_summary_table" in error_payload["message"]
    assert capture_graph.calls == 0


def test_direct_answer_route_invalidates_group_plan_and_skips_plan_execution(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    capture_graph = _CaptureGraph()
    monkeypatch.setattr(server, "graph", capture_graph)
    dataset_id = _upload_plan_execution_dataset(client)
    monkeypatch.setattr(
        chat_service,
        "plan_request_with_llm",
        lambda *args, **kwargs: intent_planner.UnifiedPlanningResult(
            routing_decision=RoutingDecision(
                primary_mode="direct_answer",
                confidence_score=0.91,
                confidence_band="high",
                needs_dataset=False,
                needs_tool_execution=False,
                needs_artifact_context=False,
                requested_capabilities=["direct_answer"],
                ambiguity_flags=[],
                guardrail_actions=[],
                fallback_reasons=[],
                reasoning_summary="用户在问概念解释。",
                execution_plan=["answer directly"],
                deliverables=["summary"],
                route_source="llm_primary",
            ),
            task_plan=TaskPlan(
                goal="错误地附带分析任务",
                planning_confidence=0.8,
                assumptions=[],
                ambiguity_flags=[],
                tasks=[
                    TaskSpec(
                        task_id="task_1",
                        task_type="group_aggregate",
                        description="按渠道分组",
                        inputs={"group_by": ["channel_source"]},
                        required_outputs=["group_summary_table"],
                    )
                ],
                final_response_style="concise_analysis",
            ),
        ),
    )

    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": "解释一下 conversion_flag 是什么意思"}]},
        },
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    route_info = next(payload for event_type, payload in events if event_type == "route_info")
    assert route_info["final_branch"] == "direct_answer"
    assert route_info["task_plan_available"] is False
    assert capture_graph.calls == 0


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
