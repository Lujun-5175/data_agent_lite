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
IntentType = Literal["analysis", "ml", "chart", "mixed", "followup", "dataset_overview"]
RouteSource = Literal["llm_primary", "llm_with_guardrail", "heuristic_fallback"]


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_mode: RoutingPrimaryMode
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_band: ConfidenceBand = "medium"
    needs_dataset: bool = False
    needs_tool_execution: bool = False
    needs_artifact_context: bool = False
    requested_capabilities: list[RoutingCapability] = Field(default_factory=list)
    ambiguity_flags: list[str] = Field(default_factory=list)
    guardrail_actions: list[str] = Field(default_factory=list)
    fallback_reasons: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    execution_plan: list[str] = Field(default_factory=list)

    # Compatibility fields retained for the existing execution flow.
    intent_type: IntentType = "analysis"
    is_dataset_overview: bool = False
    is_follow_up: bool = False
    requires_ml: bool = False
    requires_chart: bool = False
    requires_python_analysis: bool = False
    deliverables: list[str] = Field(default_factory=list)
    confidence: ConfidenceBand = "medium"
    conflict_flags: list[str] = Field(default_factory=list)
    route_source: RouteSource = "llm_primary"
    suggested_plan: list[str] = Field(default_factory=list)
