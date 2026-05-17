from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.task_plan_models import TaskPlan


VerificationStatus = Literal["success", "incomplete", "failed", "needs_clarification"]


@dataclass(slots=True)
class VerificationResult:
    status: VerificationStatus
    missing_outputs: list[str]
    failed_task_ids: list[str]
    reason: str | None = None


def verify_task_plan(*, task_plan: TaskPlan, executed_task_ids: list[str], produced_outputs: list[str]) -> VerificationResult:
    expected_task_ids = [task.task_id for task in task_plan.tasks]
    missing_task_ids = [task_id for task_id in expected_task_ids if task_id not in executed_task_ids]

    expected_outputs: list[str] = []
    for task in task_plan.tasks:
        expected_outputs.extend(str(item) for item in task.required_outputs if isinstance(item, str) and item.strip())
    produced_set = {item for item in produced_outputs if isinstance(item, str) and item.strip()}
    missing_outputs = sorted({item for item in expected_outputs if item not in produced_set})

    if not missing_task_ids and not missing_outputs:
        return VerificationResult(
            status="success",
            missing_outputs=[],
            failed_task_ids=[],
            reason=None,
        )

    reasons: list[str] = []
    if missing_task_ids:
        reasons.append(f"未完成任务：{', '.join(missing_task_ids)}")
    if missing_outputs:
        reasons.append(f"缺少输出：{', '.join(missing_outputs)}")
    return VerificationResult(
        status="incomplete",
        missing_outputs=missing_outputs,
        failed_task_ids=missing_task_ids,
        reason="；".join(reasons) if reasons else "计划未完整执行。",
    )
