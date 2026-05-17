from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.routing_models import RoutingDecision
from src.task_plan_models import TaskPlan


@dataclass(slots=True)
class LegacyRouteProjection:
    intent_type: str
    is_dataset_overview: bool
    is_follow_up: bool
    requires_ml: bool
    requires_chart: bool
    requires_python_analysis: bool
    confidence: str
    conflict_flags: list[str]
    route_source: str
    suggested_plan: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _derive_intent_type(decision: RoutingDecision) -> str:
    if decision.primary_mode == "dataset_overview":
        return "dataset_overview"
    if decision.primary_mode == "artifact_followup":
        return "followup"
    if decision.primary_mode == "visualization":
        return "chart"
    if decision.primary_mode == "modeling":
        return "ml"
    if decision.primary_mode == "mixed":
        return "mixed"
    return "analysis"


def derive_legacy_route_projection(decision: RoutingDecision) -> LegacyRouteProjection:
    requires_ml = any(
        capability in {"train_model", "evaluate_model", "feature_importance"}
        for capability in decision.requested_capabilities
    )
    requires_chart = "chart_generation" in decision.requested_capabilities
    requires_python_analysis = any(
        capability in {"group_analysis", "stat_test", "python_analysis"}
        for capability in decision.requested_capabilities
    )
    is_dataset_overview = decision.primary_mode == "dataset_overview"
    is_follow_up = decision.primary_mode == "artifact_followup" or decision.needs_artifact_context
    return LegacyRouteProjection(
        intent_type=_derive_intent_type(decision),
        is_dataset_overview=is_dataset_overview,
        is_follow_up=is_follow_up,
        requires_ml=requires_ml,
        requires_chart=requires_chart,
        requires_python_analysis=requires_python_analysis,
        confidence=decision.confidence_band,
        conflict_flags=list(decision.ambiguity_flags),
        route_source=decision.route_source,
        suggested_plan=list(decision.execution_plan),
    )


def _serialize_task_plan(task_plan: TaskPlan | None) -> dict[str, object]:
    if task_plan is None:
        return {
            "task_plan_available": False,
            "task_plan_goal": "",
            "task_plan_confidence": None,
            "task_plan_tasks": [],
            "task_plan_ambiguity_flags": [],
            "task_plan_assumptions": [],
        }
    return {
        "task_plan_available": True,
        "task_plan_goal": task_plan.goal,
        "task_plan_confidence": task_plan.planning_confidence,
        "task_plan_tasks": [
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "description": task.description,
                "depends_on": list(task.depends_on),
                "required_outputs": list(task.required_outputs),
            }
            for task in task_plan.tasks
        ],
        "task_plan_ambiguity_flags": list(task_plan.ambiguity_flags),
        "task_plan_assumptions": list(task_plan.assumptions),
        }


def _with_task_plan_status(
    payload: dict[str, object],
    *,
    task_plan_attempted: bool,
    task_plan_generation_failed: bool,
) -> dict[str, object]:
    payload["task_plan_attempted"] = task_plan_attempted
    payload["task_plan_generation_failed"] = task_plan_generation_failed
    return payload


def build_route_info_payload(
    decision: RoutingDecision,
    *,
    final_branch: str,
    task_plan: TaskPlan | None = None,
    task_plan_attempted: bool | None = None,
    task_plan_generation_failed: bool | None = None,
) -> dict[str, object]:
    legacy = derive_legacy_route_projection(decision)
    payload = {
        "primary_mode": decision.primary_mode,
        "confidence_score": decision.confidence_score,
        "intent_type": legacy.intent_type,
        "confidence": legacy.confidence,
        "route_source": legacy.route_source,
        "conflict_flags": legacy.conflict_flags,
        "ambiguity_flags": list(decision.ambiguity_flags),
        "guardrail_actions": list(decision.guardrail_actions),
        "fallback_reasons": list(decision.fallback_reasons),
        "suggested_plan": legacy.suggested_plan,
        "execution_plan": list(decision.execution_plan or legacy.suggested_plan),
        "requested_capabilities": list(decision.requested_capabilities),
        "requires_ml": legacy.requires_ml,
        "requires_chart": legacy.requires_chart,
        "requires_python_analysis": legacy.requires_python_analysis,
        "is_follow_up": legacy.is_follow_up,
        "needs_dataset": decision.needs_dataset,
        "needs_tool_execution": decision.needs_tool_execution,
        "needs_artifact_context": decision.needs_artifact_context,
        "final_branch": final_branch,
    }
    payload.update(_serialize_task_plan(task_plan))
    return _with_task_plan_status(
        payload,
        task_plan_attempted=bool(task_plan_attempted if task_plan_attempted is not None else task_plan is not None),
        task_plan_generation_failed=bool(task_plan_generation_failed),
    )


def build_route_diagnostics(
    decision: RoutingDecision,
    *,
    final_branch: str,
    task_plan: TaskPlan | None = None,
    task_plan_attempted: bool | None = None,
    task_plan_generation_failed: bool | None = None,
) -> dict[str, object]:
    legacy = derive_legacy_route_projection(decision)
    payload = {
        "primary_mode": decision.primary_mode,
        "confidence_score": decision.confidence_score,
        "final_intent": legacy.intent_type,
        "confidence": legacy.confidence,
        "conflict_flags": legacy.conflict_flags,
        "route_source": legacy.route_source,
        "used_fallback": legacy.route_source == "heuristic_fallback",
        "final_branch": final_branch,
    }
    payload.update(_serialize_task_plan(task_plan))
    return _with_task_plan_status(
        payload,
        task_plan_attempted=bool(task_plan_attempted if task_plan_attempted is not None else task_plan is not None),
        task_plan_generation_failed=bool(task_plan_generation_failed),
    )


def build_route_audit_payload(
    decision: RoutingDecision,
    *,
    final_branch: str,
    task_plan: TaskPlan | None = None,
    task_plan_attempted: bool | None = None,
    task_plan_generation_failed: bool | None = None,
) -> dict[str, object]:
    legacy = derive_legacy_route_projection(decision)
    payload = {
        "primary_mode": decision.primary_mode,
        "confidence_score": decision.confidence_score,
        "final_intent": legacy.intent_type,
        "confidence": legacy.confidence,
        "conflict_flags": legacy.conflict_flags,
        "ambiguity_flags": list(decision.ambiguity_flags),
        "guardrail_actions": list(decision.guardrail_actions),
        "fallback_reasons": list(decision.fallback_reasons),
        "requested_capabilities": list(decision.requested_capabilities),
        "needs_dataset": decision.needs_dataset,
        "needs_tool_execution": decision.needs_tool_execution,
        "needs_artifact_context": decision.needs_artifact_context,
        "route_source": legacy.route_source,
        "used_fallback": legacy.route_source == "heuristic_fallback",
        "final_branch": final_branch,
    }
    payload.update(_serialize_task_plan(task_plan))
    return _with_task_plan_status(
        payload,
        task_plan_attempted=bool(task_plan_attempted if task_plan_attempted is not None else task_plan is not None),
        task_plan_generation_failed=bool(task_plan_generation_failed),
    )
