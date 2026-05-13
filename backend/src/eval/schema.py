from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(...)
    category: str = Field(...)
    user_query: str = Field(...)
    dataset_name: str | None = None
    expected_intent: str | None = None
    accepted_intents: list[str] | None = None
    expected_tool: str | None = None
    accepted_tools: list[str] | None = None
    expected_args: dict[str, Any] | None = None
    expected_result: dict[str, Any] | None = None
    should_execute_successfully: bool | None = None
    should_be_blocked: bool | None = None
    stable_numerical_result: bool | None = None
    scoring_notes: str | None = None
    notes: str | None = None


class EvalPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(...)
    predicted_intent: str | None = None
    predicted_tool: str | None = None
    predicted_args: dict[str, Any] | None = None
    execution_status: str | None = None
    result: dict[str, Any] | None = None
    error_message: str | None = None


class EvalScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_cases: int = Field(...)
    intent_accuracy: float | None = None
    tool_accuracy: float | None = None
    argument_f1: float | None = None
    execution_success_rate: float | None = None
    blocked_request_accuracy: float | None = None
    numerical_accuracy: float | None = None
