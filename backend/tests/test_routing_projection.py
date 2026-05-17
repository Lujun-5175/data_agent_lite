from __future__ import annotations

from src.routing_models import RoutingDecision
from src.routing_projection import (
    build_route_audit_payload,
    build_route_diagnostics,
    build_route_info_payload,
    derive_legacy_route_projection,
)
from src.task_plan_models import TaskPlan, TaskSpec


def _sample_decision() -> RoutingDecision:
    return RoutingDecision(
        primary_mode="dataset_overview",
        confidence_score=0.82,
        confidence_band="high",
        needs_dataset=True,
        needs_tool_execution=False,
        needs_artifact_context=False,
        requested_capabilities=["summarize_dataset", "inspect_schema"],
        ambiguity_flags=["dataset_overview_missed"],
        guardrail_actions=["note_llm_heuristic_disagreement"],
        fallback_reasons=[],
        reasoning_summary="用户想了解当前数据集。",
        execution_plan=["summarize the dataset", "highlight schema warnings"],
        deliverables=["summary"],
        route_source="llm_primary",
    )


def _sample_task_plan() -> TaskPlan:
    return TaskPlan(
        goal="比较不同渠道的行为与转化差异",
        planning_confidence=0.76,
        assumptions=["conversion_flag 为二元列"],
        ambiguity_flags=[],
        tasks=[
            TaskSpec(
                task_id="task_1",
                task_type="group_aggregate",
                description="按 channel_source 分组汇总",
                inputs={"group_by": ["channel_source"]},
                required_outputs=["group_summary_table"],
            )
        ],
        final_response_style="concise_analysis",
    )


def test_derive_legacy_route_projection_matches_compatibility_fields():
    projection = derive_legacy_route_projection(_sample_decision())

    assert projection.intent_type == "dataset_overview"
    assert projection.is_dataset_overview is True
    assert projection.is_follow_up is False
    assert projection.requires_ml is False
    assert projection.requires_chart is False
    assert projection.requires_python_analysis is False
    assert projection.confidence == "high"
    assert projection.conflict_flags == ["dataset_overview_missed"]
    assert projection.route_source == "llm_primary"
    assert projection.suggested_plan == ["summarize the dataset", "highlight schema warnings"]


def test_build_route_info_payload_contains_new_and_legacy_route_fields():
    payload = build_route_info_payload(_sample_decision(), final_branch="dataset_overview", task_plan=_sample_task_plan())

    assert payload["primary_mode"] == "dataset_overview"
    assert payload["confidence_score"] == 0.82
    assert payload["intent_type"] == "dataset_overview"
    assert payload["confidence"] == "high"
    assert payload["route_source"] == "llm_primary"
    assert payload["conflict_flags"] == ["dataset_overview_missed"]
    assert payload["ambiguity_flags"] == ["dataset_overview_missed"]
    assert payload["requested_capabilities"] == ["summarize_dataset", "inspect_schema"]
    assert payload["requires_ml"] is False
    assert payload["needs_dataset"] is True
    assert payload["final_branch"] == "dataset_overview"
    assert payload["task_plan_available"] is True
    assert payload["task_plan_attempted"] is True
    assert payload["task_plan_generation_failed"] is False
    assert payload["task_plan_goal"] == "比较不同渠道的行为与转化差异"
    assert payload["task_plan_tasks"][0]["task_type"] == "group_aggregate"


def test_build_route_diagnostics_and_audit_payload_remain_compatible():
    diagnostics = build_route_diagnostics(_sample_decision(), final_branch="dataset_overview", task_plan=_sample_task_plan())
    audit_payload = build_route_audit_payload(_sample_decision(), final_branch="dataset_overview", task_plan=_sample_task_plan())

    assert diagnostics == {
        "primary_mode": "dataset_overview",
        "confidence_score": 0.82,
        "final_intent": "dataset_overview",
        "confidence": "high",
        "conflict_flags": ["dataset_overview_missed"],
        "route_source": "llm_primary",
        "used_fallback": False,
        "final_branch": "dataset_overview",
        "task_plan_available": True,
        "task_plan_goal": "比较不同渠道的行为与转化差异",
        "task_plan_confidence": 0.76,
        "task_plan_tasks": [
            {
                "task_id": "task_1",
                "task_type": "group_aggregate",
                "description": "按 channel_source 分组汇总",
                "depends_on": [],
                "required_outputs": ["group_summary_table"],
            }
        ],
        "task_plan_ambiguity_flags": [],
        "task_plan_assumptions": ["conversion_flag 为二元列"],
        "task_plan_attempted": True,
        "task_plan_generation_failed": False,
    }
    assert audit_payload["final_intent"] == "dataset_overview"
    assert audit_payload["route_source"] == "llm_primary"
    assert audit_payload["used_fallback"] is False
    assert audit_payload["guardrail_actions"] == ["note_llm_heuristic_disagreement"]
    assert diagnostics["task_plan_available"] is True
    assert audit_payload["task_plan_goal"] == "比较不同渠道的行为与转化差异"
