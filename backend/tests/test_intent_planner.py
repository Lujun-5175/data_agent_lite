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


def test_plan_request_with_llm_parses_routing_decision(monkeypatch: pytest.MonkeyPatch):
    planner = _FakePlannerModel(
        json.dumps(
            {
                "primary_mode": "modeling",
                "needs_dataset": True,
                "needs_tool_execution": True,
                "reasoning_summary": "User explicitly wants to train a model.",
                "execution_plan": ["train a baseline model", "report metrics"],
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", planner)

    decision = intent_planner.plan_request_with_llm("train a logistic regression model for churn")

    assert decision is not None
    assert decision.primary_mode == "modeling"
    assert decision.needs_dataset is True
    assert decision.needs_tool_execution is True
    assert decision.reasoning_summary != ""
    assert len(decision.execution_plan) >= 1
    assert len(planner.last_messages) == 1


def test_plan_request_with_llm_defaults_to_analysis_on_missing_fields(monkeypatch: pytest.MonkeyPatch):
    planner = _FakePlannerModel(json.dumps({}, ensure_ascii=False))
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", planner)

    decision = intent_planner.plan_request_with_llm("some ambiguous query")

    assert decision is not None
    assert decision.primary_mode == "analysis"
    assert decision.needs_dataset is False


def test_plan_request_with_llm_sets_needs_dataset_when_llm_says_true(monkeypatch: pytest.MonkeyPatch):
    planner = _FakePlannerModel(
        json.dumps(
            {
                "primary_mode": "analysis",
                "needs_dataset": True,
                "needs_tool_execution": True,
                "reasoning_summary": "User wants data analysis.",
                "execution_plan": ["run analysis"],
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", planner)

    decision = intent_planner.plan_request_with_llm("analyze this dataset")

    assert decision is not None
    assert decision.needs_dataset is True


def test_plan_request_with_llm_returns_none_when_model_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", None)

    decision = intent_planner.plan_request_with_llm("hello")

    assert decision is None


def test_plan_request_with_llm_accepts_suggested_plan_as_execution_plan(monkeypatch: pytest.MonkeyPatch):
    planner = _FakePlannerModel(
        json.dumps(
            {
                "primary_mode": "dataset_overview",
                "needs_dataset": True,
                "needs_tool_execution": False,
                "reasoning_summary": "User wants dataset overview.",
                "suggested_plan": ["summarize the dataset", "highlight schema warnings"],
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", planner)

    decision = intent_planner.plan_request_with_llm("explain this dataset")

    assert decision is not None
    assert decision.primary_mode == "dataset_overview"
    assert "summarize the dataset" in decision.execution_plan
