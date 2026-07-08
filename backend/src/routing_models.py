from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ConfidenceBand = Literal["low", "medium", "high"]
RoutingCapability = Literal[
    "summarize_dataset",
    "inspect_schema",
    "reuse_prior_artifact",
    "group_analysis",
    "stat_test",
    "python_analysis",
    "chart_generation",
    "train_model",
    "evaluate_model",
    "feature_importance",
    "direct_answer",
]
RoutingPrimaryMode = Literal[
    "direct_answer",
    "dataset_overview",
    "analysis",
    "visualization",
    "modeling",
    "artifact_followup",
    "mixed",
    "clarification",
]
RouteSource = Literal["llm_primary", "llm_with_guardrail", "heuristic_fallback"]

ROUTING_PRIMARY_MODES = (
    "direct_answer",
    "dataset_overview",
    "analysis",
    "visualization",
    "modeling",
    "clarification",
)


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    primary_mode: str = "analysis"
    needs_dataset: bool = False
    needs_tool_execution: bool = False
    reasoning_summary: str = ""
    execution_plan: list[str] = Field(default_factory=list)
    route_source: str = "llm"
