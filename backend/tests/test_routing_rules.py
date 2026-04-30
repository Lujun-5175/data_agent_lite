from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src import intent_planner
from src.routing_rules import (
    RoutingContext,
    decide_dataset_required,
    decide_ml_intent,
    decide_stats_intent,
    interpret_request,
)


def test_stats_intent_clear_match_with_reasons():
    decision = decide_stats_intent(
        RoutingContext(
            message="请对 MonthlyCharges 做 t-test，并按 Churn 分组比较",
            dataset_columns=["MonthlyCharges", "Churn"],
        )
    )
    assert decision.matched is True
    assert decision.score >= decision.threshold
    assert decision.reasons
    assert any("weighted keyword" in reason for reason in decision.reasons)


def test_dataset_required_clear_match():
    decision = decide_dataset_required(
        RoutingContext(message="请基于这份数据集做相关性分析")
    )
    assert decision.matched is True
    assert decision.score >= decision.threshold


def test_ambiguous_query_low_score():
    decision = decide_stats_intent(RoutingContext(message="你好，今天帮我解释一下概念"))
    assert decision.matched is False
    assert decision.score < decision.threshold


def test_schema_column_reference_boosts_score():
    base = decide_stats_intent(RoutingContext(message="帮我分析一下", dataset_columns=[]))
    boosted = decide_stats_intent(
        RoutingContext(message="帮我分析 MonthlyCharges", dataset_columns=["MonthlyCharges"])
    )
    assert boosted.score > base.score
    assert any("schema column reference" in reason for reason in boosted.reasons)


def test_prior_context_boosts_dataset_required():
    without_context = decide_dataset_required(
        RoutingContext(message="继续分析这个问题", prior_analysis_active=False)
    )
    with_context = decide_dataset_required(
        RoutingContext(message="继续分析这个问题", prior_analysis_active=True)
    )
    assert with_context.score > without_context.score
    assert any("prior analysis context" in reason for reason in with_context.reasons)


def test_route_decision_shape():
    decision = decide_dataset_required(RoutingContext(message="分析一下上传文件"))
    payload = decision.to_dict()
    assert set(payload.keys()) == {"matched", "score", "threshold", "reasons"}
    assert isinstance(payload["reasons"], list)


def test_ml_intent_clear_match():
    decision = decide_ml_intent(RoutingContext(message="请训练一个 baseline logistic regression 模型预测 churn"))
    assert decision.matched is True
    assert decision.score >= decision.threshold
    assert any("logistic regression" in reason.lower() for reason in decision.reasons)


def test_stats_query_should_not_trigger_ml():
    decision = decide_ml_intent(RoutingContext(message="请做卡方检验，查看 Contract 与 Churn 是否相关"))
    assert decision.matched is False


def test_interpretation_explicit_ml_request():
    interpretation = interpret_request(
        RoutingContext(message="train a logistic regression model for churn")
    )
    assert interpretation.intent_type == "ml"
    assert interpretation.requires_ml is True
    assert interpretation.requires_python_analysis is False
    assert "explicit ML" in interpretation.reasoning_summary
    assert interpretation.suggested_plan


def test_interpretation_exploratory_analysis_request():
    interpretation = interpret_request(
        RoutingContext(message="look at churn by contract type")
    )
    assert interpretation.intent_type == "analysis"
    assert interpretation.requires_ml is False
    assert interpretation.requires_python_analysis is True
    assert "exploratory" in interpretation.reasoning_summary.lower()


def test_interpretation_mixed_workflow_request():
    interpretation = interpret_request(
        RoutingContext(message="analyze churn drivers and, if useful, try a simple model")
    )
    assert interpretation.intent_type == "mixed"
    assert interpretation.requires_ml is True
    assert interpretation.requires_python_analysis is True
    assert interpretation.suggested_plan[0].lower().startswith("inspect") or interpretation.suggested_plan[0].lower().startswith("perform")
    assert any(any(token in step.lower() for token in ("model", "evaluate", "classification", "logistic")) for step in interpretation.suggested_plan)


def test_interpretation_follow_up_model_request():
    interpretation = interpret_request(
        RoutingContext(message="use the previous model again")
    )
    assert interpretation.intent_type == "followup"
    assert interpretation.requires_ml is True
    assert any(token in " ".join(interpretation.suggested_plan).lower() for token in ("reuse", "previous", "latest"))


class _FakeIntentPlannerModel:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_messages = None

    def invoke(self, messages, **kwargs):
        self.last_messages = messages
        return SimpleNamespace(content=self.response_text)


def test_llm_structured_intent_is_used_when_valid(monkeypatch: pytest.MonkeyPatch):
    planner = _FakeIntentPlannerModel(
        json.dumps(
            {
                "intent_type": "ml",
                "requires_ml": True,
                "requires_chart": False,
                "requires_python_analysis": False,
                "deliverables": ["metrics", "feature_importance"],
                "reasoning_summary": "用户明确要求训练模型并查看指标。",
                "suggested_plan": ["train a baseline model", "report metrics", "report feature importance"],
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", planner)

    interpretation = interpret_request(
        RoutingContext(message="train a logistic regression model for churn")
    )

    assert interpretation.intent_type == "ml"
    assert interpretation.requires_ml is True
    assert interpretation.deliverables == ["metrics", "feature_importance"]
    assert any("report metrics" in step.lower() for step in interpretation.suggested_plan)
    assert planner.last_messages is not None


def test_vague_analysis_does_not_overtrigger_ml_even_if_llm_overcalls(monkeypatch: pytest.MonkeyPatch):
    planner = _FakeIntentPlannerModel(
        json.dumps(
            {
                "intent_type": "ml",
                "requires_ml": True,
                "requires_chart": False,
                "requires_python_analysis": False,
                "deliverables": ["metrics"],
                "reasoning_summary": "模型可能有帮助。",
                "suggested_plan": ["train a model"],
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", planner)

    interpretation = interpret_request(
        RoutingContext(message="analyze churn drivers and compare groups")
    )

    assert interpretation.intent_type == "analysis"
    assert interpretation.requires_ml is False
    assert interpretation.requires_python_analysis is True
    assert any("analysis" in step.lower() or "inspect" in step.lower() for step in interpretation.suggested_plan)


def test_malformed_llm_output_falls_back_to_heuristics(monkeypatch: pytest.MonkeyPatch):
    planner = _FakeIntentPlannerModel("not-json-at-all")
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", planner)

    interpretation = interpret_request(
        RoutingContext(message="look at churn by contract type")
    )

    assert interpretation.intent_type == "analysis"
    assert interpretation.requires_ml is False
    assert interpretation.requires_python_analysis is True
