from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src import intent_planner


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
    assert decision.intent_type == "ml"
    assert decision.requires_ml is True
    assert decision.requested_capabilities == ["train_model", "evaluate_model", "feature_importance"]
    assert "metrics" in decision.deliverables
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
    assert decision.confidence == "medium"
    assert decision.intent_type == "analysis"
    assert decision.requires_python_analysis is True


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
    assert decision.intent_type == "dataset_overview"
    assert decision.requested_capabilities == ["summarize_dataset", "inspect_schema"]
    assert decision.execution_plan == ["summarize the dataset", "highlight schema warnings"]
