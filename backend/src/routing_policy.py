from __future__ import annotations

from dataclasses import dataclass, field

from src.routing_models import RoutingDecision
from src.routing_signals import is_dataset_overview_request, is_explicit_ml_request, is_follow_up_request, normalize_message_text


@dataclass(slots=True)
class RoutingPolicyResult:
    decision: RoutingDecision
    ambiguity_flags: list[str] = field(default_factory=list)
    guardrail_actions: list[str] = field(default_factory=list)
    fallback_reasons: list[str] = field(default_factory=list)
    used_fallback: bool = False
    used_guardrail: bool = False


def apply_routing_policy(
    *,
    context,
    llm_decision: RoutingDecision | None,
    heuristic_decision: RoutingDecision,
) -> RoutingPolicyResult:
    if llm_decision is None:
        fallback = heuristic_decision.model_copy(
            update={
                "route_source": "heuristic_fallback",
                "ambiguity_flags": [],
                "conflict_flags": [],
                "guardrail_actions": [],
                "fallback_reasons": ["llm_unavailable"],
            }
        )
        return RoutingPolicyResult(
            decision=fallback,
            fallback_reasons=["llm_unavailable"],
            used_fallback=True,
        )

    ambiguity_flags = _merge_unique(
        list(llm_decision.ambiguity_flags),
        list(llm_decision.conflict_flags),
        _detect_ambiguity_flags(context=context, llm_decision=llm_decision, heuristic_decision=heuristic_decision),
    )
    hard_constraint_reasons = _detect_hard_constraint_reasons(context=context, llm_decision=llm_decision)
    low_confidence = llm_decision.confidence_band == "low"
    should_fallback = bool(hard_constraint_reasons) or (low_confidence and bool(ambiguity_flags))

    if should_fallback:
        fallback_reasons = _merge_unique(
            hard_constraint_reasons,
            ["low_confidence_with_ambiguity"] if low_confidence and ambiguity_flags else [],
        )
        guardrail_actions = _build_guardrail_actions(fallback_reasons)
        fallback = heuristic_decision.model_copy(
            update={
                "route_source": "llm_with_guardrail",
                "ambiguity_flags": ambiguity_flags,
                "conflict_flags": ambiguity_flags,
                "guardrail_actions": guardrail_actions,
                "fallback_reasons": fallback_reasons,
                "reasoning_summary": _build_fallback_reasoning(
                    llm_summary=llm_decision.reasoning_summary,
                    heuristic_summary=heuristic_decision.reasoning_summary,
                    ambiguity_flags=ambiguity_flags,
                    fallback_reasons=fallback_reasons,
                ),
            }
        )
        return RoutingPolicyResult(
            decision=fallback,
            ambiguity_flags=ambiguity_flags,
            guardrail_actions=guardrail_actions,
            fallback_reasons=fallback_reasons,
            used_fallback=True,
            used_guardrail=True,
        )

    retained = llm_decision.model_copy(
        update={
            "route_source": "llm_primary",
            "ambiguity_flags": ambiguity_flags,
            "conflict_flags": ambiguity_flags,
            "guardrail_actions": [],
            "fallback_reasons": [],
        }
    )
    return RoutingPolicyResult(
        decision=retained,
        ambiguity_flags=ambiguity_flags,
        used_fallback=False,
        used_guardrail=bool(ambiguity_flags),
    )


def _detect_ambiguity_flags(*, context, llm_decision: RoutingDecision, heuristic_decision: RoutingDecision) -> list[str]:
    normalized = normalize_message_text(context.message)
    explicit_ml_requested = is_explicit_ml_request(normalized)
    follow_up_requested = is_follow_up_request(normalized, latest_artifact=context.latest_artifact)
    dataset_overview_requested = is_dataset_overview_request(normalized)

    ambiguity_flags: list[str] = []
    if dataset_overview_requested and not llm_decision.is_dataset_overview:
        ambiguity_flags.append("dataset_overview_missed")
    if follow_up_requested and not llm_decision.is_follow_up:
        ambiguity_flags.append("follow_up_missed")
    if explicit_ml_requested and not llm_decision.requires_ml:
        ambiguity_flags.append("explicit_ml_missed")
    if heuristic_decision.requires_chart and not llm_decision.requires_chart:
        ambiguity_flags.append("chart_request_missed")
    if llm_decision.requires_ml and not explicit_ml_requested and heuristic_decision.intent_type in {"analysis", "followup", "dataset_overview"}:
        ambiguity_flags.append("ml_overcall")
    if llm_decision.primary_mode == "clarification":
        ambiguity_flags.append("clarification_requested")
    return ambiguity_flags


def _detect_hard_constraint_reasons(*, context, llm_decision: RoutingDecision) -> list[str]:
    reasons: list[str] = []
    missing_artifact = context.latest_artifact is None
    if missing_artifact and (llm_decision.needs_artifact_context or llm_decision.primary_mode == "artifact_followup"):
        reasons.append("missing_artifact_context")
    return reasons


def _build_guardrail_actions(fallback_reasons: list[str]) -> list[str]:
    actions: list[str] = []
    for reason in fallback_reasons:
        if reason == "missing_artifact_context":
            actions.append("fallback_to_heuristic_due_to_missing_artifact")
        elif reason == "low_confidence_with_ambiguity":
            actions.append("fallback_to_heuristic_due_to_low_confidence")
        elif reason == "llm_unavailable":
            actions.append("fallback_to_heuristic_due_to_llm_unavailable")
    return _merge_unique(actions)


def _build_fallback_reasoning(
    *,
    llm_summary: str,
    heuristic_summary: str,
    ambiguity_flags: list[str],
    fallback_reasons: list[str],
) -> str:
    parts: list[str] = []
    if llm_summary.strip():
        parts.append(f"LLM: {llm_summary.strip()}")
    if heuristic_summary.strip():
        parts.append(f"guardrail override: {heuristic_summary.strip()}")
    if ambiguity_flags:
        parts.append(f"ambiguities: {', '.join(ambiguity_flags)}")
    if fallback_reasons:
        parts.append(f"fallback reasons: {', '.join(fallback_reasons)}")
    return "; ".join(parts) if parts else heuristic_summary


def _merge_unique(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for item in group:
            token = str(item).strip().lower()
            if token and token not in merged:
                merged.append(token)
    return merged
