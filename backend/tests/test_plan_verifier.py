from __future__ import annotations

from src.plan_verifier import verify_task_plan
from src.task_plan_models import TaskPlan, TaskSpec


def _sample_plan() -> TaskPlan:
    return TaskPlan(
        goal="按渠道汇总并输出结果",
        planning_confidence=0.8,
        assumptions=[],
        ambiguity_flags=[],
        tasks=[
            TaskSpec(
                task_id="task_1",
                task_type="group_aggregate",
                description="按渠道汇总",
                inputs={"group_by": ["channel_source"]},
                required_outputs=["group_summary_table"],
            )
        ],
        final_response_style="concise_analysis",
    )


def test_verify_task_plan_success():
    result = verify_task_plan(
        task_plan=_sample_plan(),
        executed_task_ids=["task_1"],
        produced_outputs=["group_summary_table"],
    )

    assert result.status == "success"
    assert result.missing_outputs == []
    assert result.failed_task_ids == []
    assert result.reason is None


def test_verify_task_plan_detects_missing_outputs():
    result = verify_task_plan(
        task_plan=_sample_plan(),
        executed_task_ids=["task_1"],
        produced_outputs=[],
    )

    assert result.status == "incomplete"
    assert result.missing_outputs == ["group_summary_table"]
    assert result.failed_task_ids == []
    assert "缺少输出" in (result.reason or "")
