from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src import intent_planner
from src.routing_projection import derive_legacy_route_projection
from src.task_plan_models import TaskPlan, TaskSpec


class _FakePlannerModel:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_messages = []

    def invoke(self, messages, **kwargs):
        self.last_messages.append(messages)
        return SimpleNamespace(content=self.response_text)


def test_plan_intent_with_llm_parses_routing_decision(monkeypatch: pytest.MonkeyPatch):
    planner = _FakePlannerModel(
        json.dumps(
            {
                "primary_mode": "modeling",
                "confidence_score": 0.91,
                "confidence_band": "high",
                "needs_dataset": True,
                "needs_tool_execution": True,
                "needs_artifact_context": False,
                "requested_capabilities": ["train_model", "evaluate_model", "feature_importance"],
                "ambiguity_flags": [],
                "reasoning_summary": "用户明确要求训练模型并查看指标。",
                "execution_plan": ["train a baseline model", "report metrics", "report feature importance"],
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", planner)

    decision = intent_planner.plan_intent_with_llm("train a logistic regression model for churn")

    assert decision is not None
    assert decision.primary_mode == "modeling"
    assert decision.confidence_score == pytest.approx(0.91)
    assert decision.confidence_band == "high"
    assert decision.requested_capabilities == ["train_model", "evaluate_model", "feature_importance"]
    assert "metrics" in decision.deliverables
    legacy = derive_legacy_route_projection(decision)
    assert legacy.intent_type == "ml"
    assert legacy.requires_ml is True
    assert len(planner.last_messages) == 1


def test_plan_intent_with_llm_derives_confidence_band_from_score(monkeypatch: pytest.MonkeyPatch):
    planner = _FakePlannerModel(
        json.dumps(
            {
                "primary_mode": "analysis",
                "confidence_score": 0.62,
                "needs_dataset": True,
                "needs_tool_execution": True,
                "requested_capabilities": ["group_analysis", "python_analysis"],
                "ambiguity_flags": [],
                "reasoning_summary": "用户在要求分析分组差异。",
                "execution_plan": ["inspect the dataset", "perform grouped analysis"],
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", planner)

    decision = intent_planner.plan_intent_with_llm("look at churn by contract type")

    assert decision is not None
    assert decision.confidence_band == "medium"
    legacy = derive_legacy_route_projection(decision)
    assert legacy.confidence == "medium"
    assert legacy.intent_type == "analysis"
    assert legacy.requires_python_analysis is True


def test_plan_intent_with_llm_accepts_legacy_payload_shape(monkeypatch: pytest.MonkeyPatch):
    planner = _FakePlannerModel(
        json.dumps(
            {
                "intent_type": "dataset_overview",
                "is_dataset_overview": True,
                "requires_ml": False,
                "requires_chart": False,
                "requires_python_analysis": False,
                "confidence": "high",
                "deliverables": ["summary"],
                "reasoning_summary": "用户想了解当前数据集。",
                "suggested_plan": ["summarize the dataset", "highlight schema warnings"],
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", planner)

    decision = intent_planner.plan_intent_with_llm("讲解一下这个数据集")

    assert decision is not None
    assert decision.primary_mode == "dataset_overview"
    assert decision.requested_capabilities == ["summarize_dataset", "inspect_schema"]
    assert decision.execution_plan == ["summarize the dataset", "highlight schema warnings"]
    assert derive_legacy_route_projection(decision).intent_type == "dataset_overview"


def test_plan_request_with_llm_returns_routing_and_task_plan(monkeypatch: pytest.MonkeyPatch):
    planner = _FakePlannerModel(
        json.dumps(
            {
                "routing_decision": {
                    "primary_mode": "analysis",
                    "confidence_score": 0.78,
                    "confidence_band": "high",
                    "needs_dataset": True,
                    "needs_tool_execution": True,
                    "needs_artifact_context": False,
                    "requested_capabilities": ["group_analysis", "stat_test"],
                    "ambiguity_flags": [],
                    "reasoning_summary": "用户需要比较分组差异。",
                    "execution_plan": ["group rows", "compare rates"],
                },
                "task_plan": {
                    "goal": "比较不同渠道的行为和转化差异",
                    "planning_confidence": 0.74,
                    "assumptions": ["conversion_flag 为二元列"],
                    "ambiguity_flags": [],
                    "tasks": [
                        {
                            "task_id": "task_1",
                            "task_type": "group_aggregate",
                            "description": "按 channel_source 统计均值和转化率",
                            "inputs": {"group_by": ["channel_source"]},
                            "depends_on": [],
                            "required_outputs": ["group_summary_table"],
                            "can_retry": True,
                        }
                    ],
                    "final_response_style": "concise_analysis",
                },
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", planner)

    result = intent_planner.plan_request_with_llm(
        "按 channel_source 分组，统计 session_count 均值并比较 conversion_flag 转化率",
        dataset_columns=["channel_source", "session_count", "conversion_flag"],
        dataset_summary={"row_count": 1000},
    )

    assert result is not None
    assert result.routing_decision is not None
    assert result.routing_decision.primary_mode == "analysis"
    assert result.task_plan is not None
    assert result.task_plan.goal == "比较不同渠道的行为和转化差异"
    assert result.task_plan.tasks[0].task_type == "group_aggregate"


def test_plan_request_with_llm_coerces_legacy_task_types_to_core_set(monkeypatch: pytest.MonkeyPatch):
    planner = _FakePlannerModel(
        json.dumps(
            {
                "routing_decision": {
                    "primary_mode": "analysis",
                    "confidence_score": 0.76,
                    "confidence_band": "high",
                    "needs_dataset": True,
                    "needs_tool_execution": True,
                    "needs_artifact_context": False,
                    "requested_capabilities": ["group_analysis", "stat_test"],
                    "ambiguity_flags": [],
                    "reasoning_summary": "用户需要比较分组差异并做统计检验。",
                    "execution_plan": ["group rows", "run significance test"],
                },
                "task_plan": {
                    "goal": "比较 A/B 组差异",
                    "planning_confidence": 0.7,
                    "assumptions": [],
                    "ambiguity_flags": [],
                    "tasks": [
                        {
                            "task_id": "task_1",
                            "task_type": "group_comparison",
                            "description": "比较分组转化率",
                            "inputs": {"group_by": ["ab_group"]},
                            "depends_on": [],
                            "required_outputs": ["conversion_rate_comparison"],
                            "can_retry": True,
                        },
                        {
                            "task_id": "task_2",
                            "task_type": "stat_test",
                            "description": "执行卡方检验",
                            "inputs": {"test": "chi_square"},
                            "depends_on": ["task_1"],
                            "required_outputs": ["chi_square_result"],
                            "can_retry": True,
                        },
                    ],
                    "final_response_style": "concise_analysis",
                },
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", planner)

    result = intent_planner.plan_request_with_llm(
        "比较 A/B 组差异并做卡方检验",
        dataset_columns=["ab_group", "conversion_flag"],
    )

    assert result is not None
    assert result.task_plan is not None
    assert result.task_plan.tasks[0].task_type == "group_aggregate"
    assert result.task_plan.tasks[1].task_type == "python_analysis"


def test_plan_request_with_llm_drops_unknown_task_type(monkeypatch: pytest.MonkeyPatch):
    planner = _FakePlannerModel(
        json.dumps(
            {
                "routing_decision": {
                    "primary_mode": "analysis",
                    "confidence_score": 0.7,
                    "confidence_band": "medium",
                    "needs_dataset": True,
                    "needs_tool_execution": True,
                    "needs_artifact_context": False,
                    "requested_capabilities": ["python_analysis"],
                    "ambiguity_flags": [],
                    "reasoning_summary": "用户需要分析。",
                    "execution_plan": ["run analysis"],
                },
                "task_plan": {
                    "goal": "做一次分析",
                    "planning_confidence": 0.65,
                    "assumptions": [],
                    "ambiguity_flags": [],
                    "tasks": [
                        {
                            "task_id": "task_1",
                            "task_type": "cohort_analysis",
                            "description": "未知类型任务",
                            "inputs": {},
                            "depends_on": [],
                            "required_outputs": ["result"],
                            "can_retry": True,
                        }
                    ],
                    "final_response_style": "concise_analysis",
                },
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", planner)

    result = intent_planner.plan_request_with_llm("做 cohort analysis", dataset_columns=["user_id"])

    assert result is not None
    assert result.task_plan is not None
    assert result.task_plan.tasks == []


def test_validate_task_plan_against_routing_invalidates_non_tool_plan_mismatch():
    routing_decision = intent_planner.IntentInterpretationPayload(
        primary_mode="direct_answer",
        confidence_score=0.9,
        confidence_band="high",
        needs_dataset=False,
        needs_tool_execution=False,
        needs_artifact_context=False,
        requested_capabilities=["direct_answer"],
        ambiguity_flags=[],
        guardrail_actions=[],
        fallback_reasons=[],
        reasoning_summary="直接回答即可。",
        execution_plan=["answer directly"],
        deliverables=["summary"],
        route_source="llm_primary",
    )
    task_plan = TaskPlan(
        goal="错误地附带了分析任务",
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
    )

    assert (
        intent_planner.validate_task_plan_against_routing(
            routing_decision=routing_decision,
            task_plan=task_plan,
        )
        is None
    )


def test_validate_task_plan_against_routing_keeps_supported_plan_under_guardrail_route():
    routing_decision = intent_planner.IntentInterpretationPayload(
        primary_mode="analysis",
        confidence_score=0.62,
        confidence_band="medium",
        needs_dataset=True,
        needs_tool_execution=True,
        needs_artifact_context=False,
        requested_capabilities=["group_analysis", "python_analysis"],
        ambiguity_flags=["low_confidence"],
        guardrail_actions=["fallback_to_heuristic"],
        fallback_reasons=["low_confidence_with_ambiguity"],
        reasoning_summary="已回退。",
        execution_plan=["run grouped analysis"],
        deliverables=["summary"],
        route_source="llm_with_guardrail",
    )
    task_plan = TaskPlan(
        goal="按渠道分组分析",
        planning_confidence=0.7,
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
    )

    validated = intent_planner.validate_task_plan_against_routing(
        routing_decision=routing_decision,
        task_plan=task_plan,
    )

    assert validated is not None
    assert validated.tasks[0].task_type == "group_aggregate"
