from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from src.intent_planner import IntentInterpretationPayload, plan_intent_with_llm
from src.settings import SETTINGS

logger = logging.getLogger(__name__)


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
    requires_ml: bool
    requires_chart: bool
    requires_python_analysis: bool
    deliverables: list[str]
    reasoning_summary: str
    suggested_plan: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DATASET_REQUIRED_WEIGHTS: dict[str, float] = {
    "数据集": 1.6,
    "数据": 1.2,
    "这份数据": 1.6,
    "当前数据": 1.6,
    "csv": 1.4,
    "上传": 1.2,
    "分析": 1.0,
    "相关性": 1.4,
    "分组": 1.2,
    "检验": 1.5,
    "group by": 1.3,
    "correlation": 1.4,
    "t-test": 1.5,
    "chi-square": 1.5,
}

STATS_INTENT_WEIGHTS: dict[str, float] = {
    "describe": 1.3,
    "summary": 1.2,
    "描述统计": 1.6,
    "分组汇总": 1.6,
    "group by": 1.4,
    "correlation": 1.4,
    "相关性": 1.4,
    "t-test": 1.8,
    "t检验": 1.8,
    "chi-square": 1.8,
    "卡方": 1.8,
    "anova": 1.8,
    "显著性": 1.6,
}

ML_INTENT_WEIGHTS: dict[str, float] = {
    "train model": 2.0,
    "baseline model": 1.8,
    "predict": 1.8,
    "prediction": 1.2,
    "classification": 1.8,
    "regression": 1.8,
    "logistic regression": 2.2,
    "linear regression": 2.2,
    "evaluate model": 1.8,
    "accuracy": 0.8,
    "feature importance": 1.8,
    "训练模型": 2.0,
    "预测": 1.8,
    "预测一下": 1.2,
    "分类": 1.6,
    "回归": 1.6,
    "逻辑回归": 2.2,
    "线性回归": 2.2,
    "模型评估": 1.8,
    "特征重要性": 1.8,
    "重要特征": 1.0,
}

EXPLICIT_ML_WEIGHTS: dict[str, float] = {
    "train a model": 2.4,
    "train a logistic regression model": 3.2,
    "train a linear regression model": 3.2,
    "train model": 2.4,
    "build a model": 2.4,
    "build model": 2.4,
    "try a simple model": 3.0,
    "simple model": 2.6,
    "try a model": 2.2,
    "fit model": 2.4,
    "baseline model": 2.2,
    "predict": 2.0,
    "prediction": 1.8,
    "classify": 2.2,
    "classifier": 2.2,
    "classification model": 2.2,
    "logistic regression": 2.6,
    "linear regression": 2.6,
    "evaluate model": 2.2,
    "model metrics": 2.6,
    "metrics": 2.0,
    "accuracy": 1.2,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "auc": 1.0,
    "roc auc": 1.2,
    "feature importance": 2.8,
    "coefficients": 1.6,
    "coefficient": 1.6,
    "train一个模型": 2.4,
    "训练一个模型": 2.4,
    "训练模型": 2.4,
    "训练一个 baseline": 2.2,
    "baseline model": 2.2,
    "预测": 2.0,
    "预测一下": 2.0,
    "分类器": 2.2,
    "分类模型": 2.2,
    "逻辑回归": 2.6,
    "线性回归": 2.6,
    "模型评估": 2.2,
    "模型指标": 2.6,
    "准确率": 1.2,
    "精确率": 1.0,
    "召回率": 1.0,
    "f1分数": 1.0,
    "roc auc": 1.2,
    "特征重要性": 2.8,
    "重要特征": 2.0,
}

EXPLORATORY_ANALYSIS_TERMS: dict[str, float] = {
    "analyze": 1.3,
    "analysis": 1.2,
    "explore": 1.2,
    "exploration": 1.2,
    "look at": 1.1,
    "compare": 1.2,
    "comparison": 1.1,
    "factors": 1.2,
    "drivers": 1.2,
    "influence": 1.1,
    "relationship": 1.1,
    "distribution": 1.0,
    "trend": 1.0,
    "why": 0.8,
    "reason": 0.8,
    "summary": 1.0,
    "describe": 1.0,
    "概括": 1.0,
    "分析": 1.2,
    "探索": 1.2,
    "看看": 0.8,
    "比较": 1.2,
    "因素": 1.2,
    "驱动": 1.2,
    "影响": 1.1,
    "关系": 1.1,
    "分布": 1.0,
    "趋势": 1.0,
    "原因": 0.8,
}

FOLLOW_UP_TERMS: tuple[str, ...] = (
    "解释",
    "刚才",
    "之前",
    "上一个",
    "上个",
    "这个结果",
    "这结果",
    "上面的结果",
    "继续",
    "follow up",
    "follow-up",
    "再次",
    "again",
    "reuse",
)

CHART_INTENT_TERMS: tuple[str, ...] = (
    "画图",
    "绘图",
    "生成图",
    "生成一个图",
    "图表",
    "柱状图",
    "折线图",
    "散点图",
    "直方图",
    "热力图",
    "可视化",
    "chart",
    "plot",
    "visualize",
)

DELIVERABLE_TERM_MAP: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "概括", "总结", "overview", "describe", "分析"),
    "metrics": ("metrics", "model metrics", "模型指标", "accuracy", "precision", "recall", "f1", "auc", "roc auc"),
    "feature_importance": ("feature importance", "特征重要性", "重要特征", "coefficients", "coefficient", "系数"),
    "chart": CHART_INTENT_TERMS,
    "table": ("table", "表格", "明细", "top", "top n", "前", "列表"),
    "prediction": ("predict", "prediction", "预测", "预测一下", "forecast"),
    "explanation": FOLLOW_UP_TERMS,
}

ANALYSIS_OPERATION_TERMS: dict[str, float] = {
    "group": 0.4,
    "compare": 0.4,
    "test": 0.5,
    "association": 0.5,
    "significance": 0.5,
    "分组": 0.4,
    "比较": 0.4,
    "检验": 0.5,
    "关联": 0.5,
    "显著": 0.5,
}


def _normalize(message: str) -> str:
    return re.sub(r"\s+", " ", message.strip().lower())


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
    return any(term.lower() in message for term in terms)


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
    return _contains_any(message, FOLLOW_UP_TERMS)


def _chart_score(message: str) -> bool:
    return _contains_any(message, CHART_INTENT_TERMS)


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

    if follow_up_requested and capability_count == 1:
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
        requires_ml=requires_ml,
        requires_chart=requires_chart,
        requires_python_analysis=requires_python_analysis,
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

    requires_ml = fallback_intent.requires_ml or (llm_intent.requires_ml and explicit_ml_requested)
    requires_chart = fallback_intent.requires_chart
    requires_python_analysis = fallback_intent.requires_python_analysis or (
        llm_intent.requires_python_analysis and not explicit_ml_requested and not requires_ml
    )

    deliverables = list(
        dict.fromkeys(
            [
                *fallback_intent.deliverables,
                *(deliverable for deliverable in llm_intent.deliverables if deliverable not in fallback_intent.deliverables),
            ]
        )
    )

    capability_count = sum(1 for flag in (requires_ml, requires_chart, requires_python_analysis) if flag)
    if follow_up_requested and capability_count == 1:
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
    if llm_intent.reasoning_summary.strip():
        reasoning_parts.append(f"LLM: {llm_intent.reasoning_summary.strip()}")
    if fallback_intent.reasoning_summary.strip():
        reasoning_parts.append(f"guardrail: {fallback_intent.reasoning_summary.strip()}")
    if explicit_ml_requested:
        reasoning_parts.append("explicit ML guardrail passed")
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
        requires_ml=requires_ml,
        requires_chart=requires_chart,
        requires_python_analysis=requires_python_analysis,
        deliverables=deliverables,
        reasoning_summary="; ".join(reasoning_parts),
        suggested_plan=suggested_plan,
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


def interpret_request(context: RoutingContext) -> IntentInterpretation:
    fallback_intent = _heuristic_interpret_request(context)
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
