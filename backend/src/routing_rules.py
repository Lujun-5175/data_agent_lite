from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Literal

from src.chat_config import (
    ANALYSIS_OPERATION_TERMS,
    CHART_MARKERS,
    DATASET_REQUIRED_WEIGHTS,
    DELIVERABLE_TERM_MAP,
    EXPLICIT_ML_WEIGHTS,
    EXPLORATORY_ANALYSIS_TERMS,
    FOLLOW_UP_MARKERS,
    STATS_INTENT_WEIGHTS,
    contains_any_term,
    looks_like_dataset_overview_fallback,
    normalize_message_text,
)
from src.intent_planner import IntentInterpretationPayload, plan_intent_with_llm
from src.settings import SETTINGS

logger = logging.getLogger(__name__)

RouteConfidence = Literal["low", "medium", "high"]
RouteSource = Literal["llm_primary", "llm_with_guardrail", "heuristic_fallback"]


@dataclass(slots=True)
class RoutingContext:
    message: str
    dataset_columns: list[str] = field(default_factory=list)
    prior_analysis_active: bool = False


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
    intent_type: Literal["analysis", "ml", "chart", "mixed", "followup"]
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


def _contains_any(message: str, terms: tuple[str, ...]) -> bool:
    return contains_any_term(message, terms)


def _collect_deliverables(message: str) -> list[str]:
    deliverables: list[str] = []
    for deliverable, terms in DELIVERABLE_TERM_MAP.items():
        if _contains_any(message, terms):
            deliverables.append(deliverable)
    if "table" not in deliverables and _contains_any(message, ("group by", "group", "分组", "汇总")):
        deliverables.append("table")
    if "summary" not in deliverables and _contains_any(message, ("explain", "解释", "why", "原因", "总结")):
        deliverables.append("summary")
    return list(dict.fromkeys(deliverables))


def _explicit_ml_score(message: str) -> tuple[float, list[str]]:
    return _score_weighted_terms(message, EXPLICIT_ML_WEIGHTS)


def _analysis_score(message: str) -> tuple[float, list[str]]:
    return _score_weighted_terms(message, EXPLORATORY_ANALYSIS_TERMS)


def _follow_up_score(message: str) -> bool:
    return _contains_any(message, FOLLOW_UP_MARKERS)


def _chart_score(message: str) -> bool:
    return _contains_any(message, CHART_MARKERS)


def _needs_training(message: str) -> bool:
    normalized = message
    training_terms = (
        "train a model",
        "train model",
        "build a model",
        "build model",
        "fit model",
        "baseline model",
        "predict",
        "prediction",
        "classify",
        "classifier",
        "classification model",
        "logistic regression",
        "linear regression",
        "训练模型",
        "训练一个模型",
        "训练一个 baseline",
        "预测",
        "预测一下",
        "分类器",
        "分类模型",
        "逻辑回归",
        "线性回归",
    )
    return _contains_any(normalized, training_terms)


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
    follow_up_requested = _follow_up_score(normalized)
    dataset_overview_requested = looks_like_dataset_overview_fallback(normalized)
    stats_decision = decide_stats_intent(context)
    ml_decision = decide_ml_intent(context)

    follow_up_model_hint = follow_up_requested and _contains_any(normalized, ("model", "模型", "指标", "metrics", "feature importance", "重要特征"))
    requires_ml = ml_decision.matched or explicit_ml_score >= SETTINGS.routing_ml_intent_threshold or follow_up_model_hint
    requires_chart = chart_requested
    requires_python_analysis = (
        stats_decision.matched
        or analysis_score >= 1.0
        or any(token in normalized for token in ("explore", "analyze", "analysis", "compare", "factors", "drivers", "relationship", "distribution", "trend"))
    )

    capability_count = sum(1 for flag in (requires_ml, requires_chart, requires_python_analysis) if flag)

    if dataset_overview_requested:
        intent_type: Literal["analysis", "ml", "chart", "mixed", "followup"] = "followup"
    elif follow_up_requested and capability_count == 1:
        intent_type: Literal["analysis", "ml", "chart", "mixed", "followup"] = "followup"
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
    if intent_type == "followup":
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
    normalized = _normalize(context.message)
    if not normalized:
        return False
    explicit_score, _ = _explicit_ml_score(normalized)
    follow_up_model_hint = _follow_up_score(normalized) and _contains_any(
        normalized,
        ("model", "模型", "指标", "metrics", "feature importance", "重要特征"),
    )
    return explicit_score >= SETTINGS.routing_ml_intent_threshold or follow_up_model_hint


def _merge_llm_and_heuristic_intent(
    context: RoutingContext,
    llm_intent: IntentInterpretationPayload,
    fallback_intent: IntentInterpretation,
) -> IntentInterpretation:
    normalized = _normalize(context.message)
    explicit_ml_requested = _looks_like_explicit_ml_request(context)
    follow_up_requested = _follow_up_score(normalized)
    dataset_overview_requested = looks_like_dataset_overview_fallback(normalized)
    confidence = _normalize_confidence(llm_intent.confidence)
    conflict_flags = _merge_conflict_flags(
        _detect_conflict_flags(
            llm_intent=llm_intent,
            fallback_intent=fallback_intent,
            explicit_ml_requested=explicit_ml_requested,
            follow_up_requested=follow_up_requested,
            dataset_overview_requested=dataset_overview_requested,
        ),
        llm_intent.conflict_flags,
    )
    guardrail_override = confidence == "low" and bool(conflict_flags)
    route_source: RouteSource = "llm_with_guardrail" if guardrail_override else "llm_primary"

    if guardrail_override:
        return _build_guardrailed_interpretation(
            llm_intent=llm_intent,
            fallback_intent=fallback_intent,
            conflict_flags=conflict_flags,
            route_source=route_source,
        )

    is_dataset_overview = bool(llm_intent.is_dataset_overview)
    is_follow_up = bool(llm_intent.is_follow_up)
    requires_ml = bool(llm_intent.requires_ml)
    requires_chart = bool(llm_intent.requires_chart)
    requires_python_analysis = bool(llm_intent.requires_python_analysis or is_dataset_overview)
    deliverables = list(dict.fromkeys(llm_intent.deliverables or fallback_intent.deliverables))

    intent_type = _resolve_intent_type(
        is_dataset_overview=is_dataset_overview,
        is_follow_up=is_follow_up,
        follow_up_requested=follow_up_requested,
        requires_ml=requires_ml,
        requires_chart=requires_chart,
        requires_python_analysis=requires_python_analysis,
    )

    reasoning_parts: list[str] = []
    if llm_intent.reasoning_summary.strip():
        reasoning_parts.append(f"LLM: {llm_intent.reasoning_summary.strip()}")
    if conflict_flags and fallback_intent.reasoning_summary.strip():
        reasoning_parts.append(f"guardrail: {fallback_intent.reasoning_summary.strip()}")
    if conflict_flags:
        reasoning_parts.append(f"conflicts: {', '.join(conflict_flags)}")
    if not reasoning_parts:
        reasoning_parts.append("defaulted to exploratory analysis")

    suggested_plan = _merge_suggested_plan(
        llm_intent.suggested_plan or fallback_intent.suggested_plan,
        requires_ml=requires_ml,
        requires_chart=requires_chart,
        requires_python_analysis=requires_python_analysis,
        deliverables=deliverables,
        follow_up_requested=follow_up_requested,
    )

    return IntentInterpretation(
        intent_type=intent_type,
        is_dataset_overview=is_dataset_overview,
        is_follow_up=is_follow_up,
        requires_ml=requires_ml,
        requires_chart=requires_chart,
        requires_python_analysis=requires_python_analysis,
        confidence=confidence,
        conflict_flags=conflict_flags,
        route_source=route_source,
        deliverables=deliverables,
        reasoning_summary="; ".join(reasoning_parts),
        suggested_plan=suggested_plan,
    )


def _normalize_confidence(value: str | None) -> RouteConfidence:
    if value == "high":
        return "high"
    if value == "low":
        return "low"
    return "medium"


def _merge_conflict_flags(*values: list[str]) -> list[str]:
    merged: list[str] = []
    for group in values:
        for value in group:
            token = value.strip().lower()
            if token and token not in merged:
                merged.append(token)
    return merged


def _detect_conflict_flags(
    *,
    llm_intent: IntentInterpretationPayload,
    fallback_intent: IntentInterpretation,
    explicit_ml_requested: bool,
    follow_up_requested: bool,
    dataset_overview_requested: bool,
) -> list[str]:
    conflict_flags: list[str] = []
    if dataset_overview_requested and not llm_intent.is_dataset_overview:
        conflict_flags.append("dataset_overview_missed")
    if follow_up_requested and not llm_intent.is_follow_up:
        conflict_flags.append("follow_up_missed")
    if explicit_ml_requested and not llm_intent.requires_ml:
        conflict_flags.append("explicit_ml_missed")
    if fallback_intent.requires_chart and not llm_intent.requires_chart:
        conflict_flags.append("chart_request_missed")
    if llm_intent.requires_ml and not explicit_ml_requested and fallback_intent.intent_type in {"analysis", "followup"}:
        conflict_flags.append("ml_overcall")
    return conflict_flags


def _resolve_intent_type(
    *,
    is_dataset_overview: bool,
    is_follow_up: bool,
    follow_up_requested: bool,
    requires_ml: bool,
    requires_chart: bool,
    requires_python_analysis: bool,
) -> Literal["analysis", "ml", "chart", "mixed", "followup"]:
    capability_count = sum(1 for flag in (requires_ml, requires_chart, requires_python_analysis) if flag)
    if is_dataset_overview:
        return "followup"
    if is_follow_up and capability_count == 1:
        return "followup"
    if capability_count > 1:
        return "mixed"
    if requires_ml:
        return "ml"
    if requires_chart:
        return "chart"
    if requires_python_analysis:
        return "analysis"
    if follow_up_requested:
        return "followup"
    return "analysis"


def _build_guardrailed_interpretation(
    *,
    llm_intent: IntentInterpretationPayload,
    fallback_intent: IntentInterpretation,
    conflict_flags: list[str],
    route_source: RouteSource,
) -> IntentInterpretation:
    reasoning_parts: list[str] = []
    if llm_intent.reasoning_summary.strip():
        reasoning_parts.append(f"LLM: {llm_intent.reasoning_summary.strip()}")
    if fallback_intent.reasoning_summary.strip():
        reasoning_parts.append(f"guardrail override: {fallback_intent.reasoning_summary.strip()}")
    if conflict_flags:
        reasoning_parts.append(f"conflicts: {', '.join(conflict_flags)}")
    return IntentInterpretation(
        intent_type=fallback_intent.intent_type,
        is_dataset_overview=fallback_intent.is_dataset_overview,
        is_follow_up=fallback_intent.is_follow_up,
        requires_ml=fallback_intent.requires_ml,
        requires_chart=fallback_intent.requires_chart,
        requires_python_analysis=fallback_intent.requires_python_analysis,
        confidence="low",
        conflict_flags=conflict_flags,
        route_source=route_source,
        deliverables=fallback_intent.deliverables,
        reasoning_summary="; ".join(reasoning_parts) if reasoning_parts else fallback_intent.reasoning_summary,
        suggested_plan=_merge_suggested_plan(
            fallback_intent.suggested_plan,
            requires_ml=fallback_intent.requires_ml,
            requires_chart=fallback_intent.requires_chart,
            requires_python_analysis=fallback_intent.requires_python_analysis,
            deliverables=fallback_intent.deliverables,
            follow_up_requested=fallback_intent.is_follow_up,
        ),
    )


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


def interpret_request(context: RoutingContext, *, use_llm: bool = True) -> IntentInterpretation:
    fallback_intent = _heuristic_interpret_request(context)
    if not use_llm:
        return fallback_intent
    llm_intent = plan_intent_with_llm(
        context.message,
        dataset_columns=context.dataset_columns,
        prior_analysis_active=context.prior_analysis_active,
    )
    if llm_intent is None:
        return fallback_intent
    try:
        return _merge_llm_and_heuristic_intent(context, llm_intent, fallback_intent)
    except Exception:  # pragma: no cover - guardrail fallback
        logger.exception("Failed to merge LLM and heuristic intent; falling back to heuristic interpretation.")
        return fallback_intent
