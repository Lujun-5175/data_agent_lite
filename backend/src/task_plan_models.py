from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TaskType = Literal[
    "dataset_summary",
    "group_aggregate",
    "artifact_lookup",
    "python_analysis",
    "model_train",
    "direct_answer",
    "clarification",
]


class TaskMetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    agg: str
    semantic: str | None = None


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_type: TaskType
    description: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    required_outputs: list[str] = Field(default_factory=list)
    can_retry: bool = True


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    planning_confidence: float = Field(ge=0.0, le=1.0)
    assumptions: list[str] = Field(default_factory=list)
    ambiguity_flags: list[str] = Field(default_factory=list)
    tasks: list[TaskSpec] = Field(default_factory=list)
    final_response_style: str | None = None
