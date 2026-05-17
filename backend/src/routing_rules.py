from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Literal

from src.intent_planner import IntentInterpretationPayload, plan_intent_with_llm
from src.routing_models import RoutingDecision
from src.routing_policy import apply_routing_policy
from src.routing_signals import (
    ANALYSIS_OPERATION_TERMS,
    DATASET_REQUIRED_WEIGHTS,
    EXPLICIT_ML_WEIGHTS,
    EXPLORATORY_ANALYSIS_TERMS,
    STATS_INTENT_WEIGHTS,
    collect_deliverables,
    contains_any_term,
    is_chart_request,
    is_dataset_overview_request,
    is_explicit_ml_request,
    is_follow_up_request,
    normalize_message_text,
)
from src.settings import SETTINGS

logger = logging.getLogger(__name__)

RouteConfidence = Literal["low", "medium", "high"]
RouteSource = Literal["llm_primary", "llm_with_guardrail", "heuristic_fallback"]
IntentType = Literal["analysis", "ml", "chart", "mixed", "followup", "dataset_overview"]


@dataclass(slots=True)
class RoutingContext:
    message: str
    dataset_columns: list[str] = field(default_factory=list)
    prior_analysis_active: bool = False
    latest_artifact: dict[str, object] | None = None


@dataclass(slots=True)
class RouteDecision:
    matched: bool
    score: float
    threshold: float
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class IntentInterpretation:
    intent_type: IntentType
    is_dataset_overview: bool
    is_follow_up: bool
    requires_ml: bool
    requires_chart: bool
    requires_python_analysis: bool
    confidence: RouteConfidence
    conflict_flags: list[str]
    route_source: RouteSource
    deliverables: list[str]
    reasoning_summary: str
    suggested_plan: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize(message: str) -> str:
    return normalize_message_text(message)


def _score_weighted_terms(message: str, weights: dict[str, float]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    for phrase, weight in weights.items():
        if phrase in message:
            score += weight
            reasons.append(f"matched weighted keyword: {phrase} (+{weight:.1f})")
    return score, reasons


def _schema_boost(message: str, dataset_columns: list[str]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    lowered_columns = [col.strip().lower() for col in dataset_columns if col.strip()]
    for column in lowered_columns:
        if column and column in message:
            score += 0.8
            reasons.append(f"matched schema column reference: {column} (+0.8)")
    return score, reasons


def _context_boost(prior_analysis_active: bool) -> tuple[float, list[str]]:
    if prior_analysis_active:
        return 0.7, ["matched prior analysis context (+0.7)"]
    return 0.0, []


def _operation_boost(message: str) -> tuple[float, list[str]]:
    score, reasons = _score_weighted_terms(message, ANALYSIS_OPERATION_TERMS)
    mapped = [reason.replace("weighted keyword", "analysis operator") for reason in reasons]
    return score, mapped


def _collect_deliverables(message: str) -> list[str]:
    return collect_deliverables(message)


def _explicit_ml_score(message: str) -> tuple[float, list[str]]:
    return _score_weighted_terms(message, EXPLICIT_ML_WEIGHTS)


def _analysis_score(message: str) -> tuple[float, list[str]]:
    return _score_weighted_terms(message, EXPLORATORY_ANALYSIS_TERMS)


def _follow_up_score(message: str) -> bool:
    return is_follow_up_request(message)


def _chart_score(message: str) -> bool:
    return is_chart_request(message)


def decide_dataset_required(context: RoutingContext) -> RouteDecision:
    normalized = _normalize(context.message)
    if not normalized:
        return RouteDecision(matched=False, score=0.0, threshold=SETTINGS.routing_dataset_required_threshold, reasons=[])

    score = 0.0
    reasons: list[str] = []

    delta, reason_items = _score_weighted_terms(normalized, DATASET_REQUIRED_WEIGHTS)
    score += delta
    reasons.extend(reason_items)

    delta, reason_items = _schema_boost(normalized, context.dataset_columns)
    score += delta
    reasons.extend(reason_items)

    delta, reason_items = _context_boost(context.prior_analysis_active)
    score += delta
    reasons.extend(reason_items)

    matched = score >= SETTINGS.routing_dataset_required_threshold
    return RouteDecision(
        matched=matched,
        score=round(score, 3),
        threshold=SETTINGS.routing_dataset_required_threshold,
        reasons=reasons,
    )


def decide_stats_intent(context: RoutingContext) -> RouteDecision:
    normalized = _normalize(context.message)
    if not normalized:
        return RouteDecision(matched=False, score=0.0, threshold=SETTINGS.routing_stats_intent_threshold, reasons=[])

    score = 0.0
    reasons: list[str] = []

    delta, reason_items = _score_weighted_terms(normalized, STATS_INTENT_WEIGHTS)
    score += delta
    reasons.extend(reason_items)

    delta, reason_items = _operation_boost(normalized)
    score += delta
    reasons.extend(reason_items)

    delta, reason_items = _schema_boost(normalized, context.dataset_columns)
    score += delta
    reasons.extend(reason_items)

    delta, reason_items = _context_boost(context.prior_analysis_active)
    score += delta
    reasons.extend(reason_items)

    matched = score >= SETTINGS.routing_stats_intent_threshold
    return RouteDecision(
        matched=matched,
        score=round(score, 3),
        threshold=SETTINGS.routing_stats_intent_threshold,
        reasons=reasons,
    )


def decide_ml_intent(context: RoutingContext) -> RouteDecision:
    normalized = _normalize(context.message)
    if not normalized:
        return RouteDecision(matched=False, score=0.0, threshold=SETTINGS.routing_ml_intent_threshold, reasons=[])

    score = 0.0
    reasons: list[str] = []

    delta, reason_items = _explicit_ml_score(normalized)
    score += delta
    reasons.extend(reason_items)

    delta, reason_items = _schema_boost(normalized, context.dataset_columns)
    score += delta
    reasons.extend(reason_items)

    delta, reason_items = _context_boost(context.prior_analysis_active)
    score += delta
    reasons.extend(reason_items)

    # Keep ML routing conservative: explicit ML language should dominate.
    if "统计" in normalized or "t-test" in normalized or "卡方" in normalized or "anova" in normalized:
        score -= 0.8
        reasons.append("detected stats-specific terms (-0.8)")

    matched = score >= SETTINGS.routing_ml_intent_threshold
    return RouteDecision(
        matched=matched,
        score=round(score, 3),
        threshold=SETTINGS.routing_ml_intent_threshold,
        reasons=reasons,
    )


def _heuristic_interpret_request(context: RoutingContext) -> IntentInterpretation:
    normalized = _normalize(context.message)
    deliverables = _collect_deliverables(normalized)
    explicit_ml_score, explicit_ml_reasons = _explicit_ml_score(normalized)
    analysis_score, analysis_reasons = _analysis_score(normalized)
    chart_requested = _chart_score(normalized)
    follow_up_requested = is_follow_up_request(normalized, latest_artifact=context.latest_artifact)
    dataset_overview_requested = is_dataset_overview_request(normalized)
    stats_decision = decide_stats_intent(context)
    ml_decision = decide_ml_intent(context)

    follow_up_model_hint = follow_up_requested and any(
        token in normalized for token in ("model", "模型", "指标", "metrics", "feature importance", "重要特征")
    )
    requires_ml = ml_decision.matched or explicit_ml_score >= SETTINGS.routing_ml_intent_threshold or follow_up_model_hint
    requires_chart = chart_requested
    requires_python_analysis = (
        stats_decision.matched
        or analysis_score >= 1.0
        or any(token in normalized for token in ("explore", "analyze", "analysis", "compare", "factors", "drivers", "relationship", "distribution", "trend"))
    )

    capability_count = sum(1 for flag in (requires_ml, requires_chart, requires_python_analysis) if flag)

    if dataset_overview_requested:
        intent_type: IntentType = "dataset_overview"
    elif follow_up_requested and capability_count == 1:
        intent_type = "followup"
    elif capability_count > 1:
        intent_type = "mixed"
    elif requires_ml:
        intent_type = "ml"
    elif requires_chart:
        intent_type = "chart"
    elif requires_python_analysis:
        intent_type = "analysis"
    elif follow_up_requested:
        intent_type = "followup"
    else:
        intent_type = "analysis"

    reasoning_parts: list[str] = []
    if explicit_ml_reasons:
        reasoning_parts.append("explicit ML signals detected")
    if analysis_reasons:
        reasoning_parts.append("exploratory analysis signals detected")
    if chart_requested:
        reasoning_parts.append("chart request detected")
    if follow_up_requested:
        reasoning_parts.append("follow-up language detected")
    if dataset_overview_requested:
        reasoning_parts.append("dataset overview request detected")
    if follow_up_model_hint:
        reasoning_parts.append("follow-up model reuse hint detected")
    if not reasoning_parts:
        reasoning_parts.append("defaulted to exploratory analysis")

    suggested_plan: list[str] = []
    if intent_type == "dataset_overview":
        suggested_plan.append("inspect the current dataset schema profile and preprocessing context")
        suggested_plan.append("summarize the dataset shape, field groups, warnings, and suggested next questions")
    elif intent_type == "followup":
        suggested_plan.append("resolve the referenced prior result or artifact")
        suggested_plan.append("reuse the latest relevant artifact if available")
    else:
        if requires_python_analysis:
            suggested_plan.append("inspect the dataset or prior result")
            suggested_plan.append("perform the smallest necessary analysis step")
        if requires_chart:
            suggested_plan.append("generate the requested chart from the analyzed data")
        if requires_ml:
            if requires_python_analysis:
                suggested_plan.append("if the exploratory findings justify it, train a baseline model")
            else:
                suggested_plan.append("train the explicitly requested baseline model")
            if "metrics" in deliverables:
                suggested_plan.append("report model metrics")
            if "feature_importance" in deliverables:
                suggested_plan.append("report feature importance")
        if not suggested_plan:
            suggested_plan.append("answer with grounded exploratory analysis")

    return IntentInterpretation(
        intent_type=intent_type,
        is_dataset_overview=dataset_overview_requested,
        is_follow_up=follow_up_requested,
        requires_ml=requires_ml,
        requires_chart=requires_chart,
        requires_python_analysis=requires_python_analysis,
        confidence="low",
        conflict_flags=[],
        route_source="heuristic_fallback",
        deliverables=deliverables,
        reasoning_summary="; ".join(reasoning_parts),
        suggested_plan=suggested_plan,
    )


def _looks_like_explicit_ml_request(context: RoutingContext) -> bool:
    return is_explicit_ml_request(context.message)


def _routing_decision_to_intent_interpretation(decision: RoutingDecision) -> IntentInterpretation:
    return IntentInterpretation(
        intent_type=decision.intent_type,
        is_dataset_overview=decision.is_dataset_overview,
        is_follow_up=decision.is_follow_up,
        requires_ml=decision.requires_ml,
        requires_chart=decision.requires_chart,
        requires_python_analysis=decision.requires_python_analysis,
        confidence=decision.confidence,
        conflict_flags=list(decision.conflict_flags),
        route_source=decision.route_source,
        deliverables=list(decision.deliverables),
        reasoning_summary=decision.reasoning_summary,
        suggested_plan=list(decision.suggested_plan),
    )


def _normalize_confidence(value: str | None) -> RouteConfidence:
    if value == "high":
        return "high"
    if value == "low":
        return "low"
    return "medium"


def _merge_suggested_plan(
    plan: list[str],
    *,
    requires_ml: bool,
    requires_chart: bool,
    requires_python_analysis: bool,
    deliverables: list[str],
    follow_up_requested: bool,
) -> list[str]:
    merged = [step.strip() for step in plan if isinstance(step, str) and step.strip()]

    if follow_up_requested and not any(keyword in " ".join(merged).lower() for keyword in ("resolve", "reuse", "previous", "latest")):
        merged.insert(0, "resolve the referenced prior result or artifact")
        merged.insert(1, "reuse the latest relevant artifact if available")

    if requires_python_analysis and not any(keyword in " ".join(merged).lower() for keyword in ("inspect", "analyze", "analysis", "compare", "group", "filter")):
        merged.insert(0, "inspect the dataset or prior result")

    if requires_chart and not any(keyword in " ".join(merged).lower() for keyword in ("chart", "plot", "visual")):
        merged.append("generate the requested chart from the analyzed data")

    if requires_ml and not any(keyword in " ".join(merged).lower() for keyword in ("train", "metrics", "feature importance", "importance", "evaluate")):
        if requires_python_analysis:
            merged.append("if the exploratory findings justify it, train a baseline model")
        else:
            merged.append("train the explicitly requested baseline model")

    if "metrics" in deliverables and not any("metric" in step.lower() for step in merged):
        merged.append("report model metrics")
    if "feature_importance" in deliverables and not any("feature" in step.lower() or "importance" in step.lower() for step in merged):
        merged.append("report feature importance")
    if not merged:
        merged.append("answer with grounded exploratory analysis")
    return list(dict.fromkeys(merged))


def _heuristic_interpretation_to_routing_decision(interpretation: IntentInterpretation) -> RoutingDecision:
    return RoutingDecision(
        primary_mode=(
            "dataset_overview"
            if interpretation.intent_type == "dataset_overview"
            else "artifact_followup"
            if interpretation.intent_type == "followup"
            else "visualization"
            if interpretation.intent_type == "chart"
            else "modeling"
            if interpretation.intent_type == "ml"
            else "mixed"
            if interpretation.intent_type == "mixed"
            else "analysis"
        ),
        confidence_score=0.3,
        confidence_band=_normalize_confidence(interpretation.confidence),
        needs_dataset=interpretation.intent_type != "analysis" or interpretation.requires_python_analysis or interpretation.requires_chart or interpretation.requires_ml,
        needs_tool_execution=interpretation.requires_python_analysis or interpretation.requires_chart or interpretation.requires_ml,
        needs_artifact_context=interpretation.is_follow_up,
        requested_capabilities=[],
        ambiguity_flags=list(interpretation.conflict_flags),
        reasoning_summary=interpretation.reasoning_summary,
        execution_plan=list(interpretation.suggested_plan),
        intent_type=interpretation.intent_type,
        is_dataset_overview=interpretation.is_dataset_overview,
        is_follow_up=interpretation.is_follow_up,
        requires_ml=interpretation.requires_ml,
        requires_chart=interpretation.requires_chart,
        requires_python_analysis=interpretation.requires_python_analysis,
        deliverables=list(interpretation.deliverables),
        confidence=_normalize_confidence(interpretation.confidence),
        conflict_flags=list(interpretation.conflict_flags),
        route_source=interpretation.route_source,
        suggested_plan=list(interpretation.suggested_plan),
    )


def interpret_request_decision(context: RoutingContext, *, use_llm: bool = True) -> RoutingDecision:
    heuristic_interpretation = _heuristic_interpret_request(context)
    heuristic_decision = _heuristic_interpretation_to_routing_decision(heuristic_interpretation)
    if not use_llm:
        return heuristic_decision

    llm_decision = plan_intent_with_llm(
        context.message,
        dataset_columns=context.dataset_columns,
        prior_analysis_active=context.prior_analysis_active,
    )
    policy_result = apply_routing_policy(
        context=context,
        llm_decision=llm_decision,
        heuristic_decision=heuristic_decision,
    )
    final_decision = policy_result.decision
    final_plan = _merge_suggested_plan(
        final_decision.suggested_plan or final_decision.execution_plan or heuristic_interpretation.suggested_plan,
        requires_ml=final_decision.requires_ml,
        requires_chart=final_decision.requires_chart,
        requires_python_analysis=final_decision.requires_python_analysis,
        deliverables=final_decision.deliverables or heuristic_interpretation.deliverables,
        follow_up_requested=final_decision.is_follow_up,
    )
    return final_decision.model_copy(
        update={
            "confidence": _normalize_confidence(final_decision.confidence_band),
            "suggested_plan": final_plan,
            "execution_plan": final_plan,
        }
    )


def interpret_request(context: RoutingContext, *, use_llm: bool = True) -> IntentInterpretation:
    try:
        decision = interpret_request_decision(context, use_llm=use_llm)
        return _routing_decision_to_intent_interpretation(decision)
    except Exception:  # pragma: no cover - guardrail fallback
        logger.exception("Failed to merge LLM and heuristic intent; falling back to heuristic interpretation.")
        return _heuristic_interpret_request(context)
