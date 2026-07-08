"""
Minimal routing rules — restored from deleted module to fix imports.
These will be consolidated in Phase 3 refactoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RoutingContext:
    message: str = ""
    dataset_columns: list[str] = field(default_factory=list)
    prior_analysis_active: bool = False
    latest_artifact: dict[str, object] | None = None


@dataclass(slots=True)
class RouteDecision:
    matched: bool = False
    score: float = 0.0
    threshold: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"matched": self.matched, "score": self.score, "threshold": self.threshold, "reasons": self.reasons}


def decide_dataset_required(context: RoutingContext) -> RouteDecision:
    return RouteDecision(matched=True, score=5.0, threshold=3.0, reasons=["dataset_required_fallback"])


def decide_stats_intent(context: RoutingContext) -> RouteDecision:
    return RouteDecision()


def decide_ml_intent(context: RoutingContext) -> RouteDecision:
    return RouteDecision()
