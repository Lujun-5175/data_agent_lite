from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data_manager import get_data_context_summary, get_dataframe
from src.errors import AppError
from src.task_plan_models import TaskPlan, TaskSpec
from src.tools import SafeExecutionError, StatsHelperAPI


SUPPORTED_TASK_TYPES = {"dataset_summary", "group_aggregate"}


@dataclass(slots=True)
class PlanExecutionResult:
    content: str
    executed_task_ids: list[str]
    produced_outputs: list[str]


def build_executable_task_plan(task_plan: TaskPlan | None) -> TaskPlan | None:
    if task_plan is None or not task_plan.tasks:
        return None

    executable_tasks: list[TaskSpec] = []
    executable_task_ids: set[str] = set()
    for task in task_plan.tasks:
        if not _supports_task(task):
            continue
        if any(depends_on not in executable_task_ids for depends_on in task.depends_on):
            continue
        executable_tasks.append(task)
        executable_task_ids.add(task.task_id)

    if not executable_tasks:
        return None

    return TaskPlan.model_validate(
        {
            "goal": task_plan.goal,
            "planning_confidence": task_plan.planning_confidence,
            "assumptions": list(task_plan.assumptions),
            "ambiguity_flags": list(task_plan.ambiguity_flags),
            "tasks": executable_tasks,
            "final_response_style": task_plan.final_response_style,
        }
    )


def supports_task_plan(task_plan: TaskPlan | None) -> bool:
    return build_executable_task_plan(task_plan) is not None


def execute_task_plan(*, dataset_id: str, task_plan: TaskPlan) -> PlanExecutionResult:
    executable_task_plan = build_executable_task_plan(task_plan)
    if executable_task_plan is None:
        raise AppError(
            "unsupported_task_plan",
            "当前结构化计划包含暂不支持的任务类型，已回退到通用分析执行。",
            200,
            stage="plan_execution",
        )

    sections: list[str] = []
    executed_task_ids: list[str] = []
    produced_outputs: list[str] = []
    for task in executable_task_plan.tasks:
        if task.task_type == "dataset_summary":
            sections.append(_execute_dataset_summary(dataset_id))
        elif task.task_type == "group_aggregate":
            content, outputs = _execute_group_aggregate(dataset_id, task)
            sections.append(content)
            produced_outputs.extend(outputs)
        else:  # pragma: no cover - guarded by supports_task_plan
            raise AppError(
                "unsupported_task_plan",
                f"暂不支持的任务类型：{task.task_type}",
                200,
                stage="plan_execution",
            )
        executed_task_ids.append(task.task_id)

    skipped_task_count = max(0, len(task_plan.tasks) - len(executable_task_plan.tasks))
    header = [f"已按统一计划执行：{executable_task_plan.goal}"]
    if skipped_task_count:
        header.extend(
            [
                "",
                f"说明：当前先执行了 {len(executable_task_plan.tasks)} 个可直接落地的计划任务，"
                f"另有 {skipped_task_count} 个更复杂任务暂未走结构化计划分支。",
            ]
        )
    if executable_task_plan.assumptions:
        header.append("")
        header.append("执行假设：")
        header.extend(f"- {item}" for item in executable_task_plan.assumptions)
    return PlanExecutionResult(
        content="\n".join([*header, "", *sections]).strip(),
        executed_task_ids=executed_task_ids,
        produced_outputs=sorted({item for item in produced_outputs if item}),
    )


def _supports_task(task: TaskSpec) -> bool:
    if task.task_type not in SUPPORTED_TASK_TYPES:
        return False
    if task.task_type == "group_aggregate":
        group_by = task.inputs.get("group_by")
        if not isinstance(group_by, list) or len(group_by) != 1:
            return False
    return True


def _execute_dataset_summary(dataset_id: str) -> str:
    summary = get_data_context_summary(dataset_id)
    lines = [
        "数据集概况：",
        f"- 数据规模：{summary['row_count']:,} 行 × {summary['column_count']:,} 列",
        f"- 数值字段数：{summary['numeric_column_count']}",
        f"- 分类字段数：{summary['categorical_column_count']}",
    ]
    warnings = summary.get("warnings") or []
    if warnings:
        lines.append("- 需要注意：")
        lines.extend(f"  - {warning}" for warning in warnings[:3])
    return "\n".join(lines)


def _execute_group_aggregate(dataset_id: str, task: TaskSpec) -> tuple[str, list[str]]:
    group_by_values = task.inputs.get("group_by") or []
    group_by = str(group_by_values[0]).strip()
    if not group_by:
        raise AppError(
            "invalid_task_plan",
            "group_aggregate 缺少 group_by。",
            200,
            stage="plan_execution",
        )

    metric_specs = _build_group_metrics(task.inputs.get("metrics"))
    df = get_dataframe(dataset_id)
    stats = StatsHelperAPI(df, dataset_id=dataset_id)
    try:
        artifact = stats.group_summary(group_by=group_by, metrics=metric_specs, top_n=None)
    except SafeExecutionError as exc:
        raise AppError(
            "task_plan_execution_error",
            str(exc),
            200,
            stage="plan_execution",
        ) from exc

    rows = artifact.get("rows", []) if isinstance(artifact, dict) else []
    warnings = artifact.get("warnings", []) if isinstance(artifact, dict) else []
    rate_metadata = artifact.get("rate_metadata", []) if isinstance(artifact, dict) else []

    lines = [
        f"任务 `{task.task_id}`：{task.description}",
        "",
        _render_markdown_table(rows),
    ]
    if rate_metadata:
        lines.append("")
        lines.append("转化率说明：")
        for item in rate_metadata:
            metric_name = item.get("metric")
            source_column = item.get("source_column")
            positive_label = item.get("positive_label")
            lines.append(
                f"- `{metric_name}` 基于列 `{source_column}` 计算，正类取值为 `{positive_label}`。"
            )
    if warnings:
        lines.append("")
        lines.append("注意事项：")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines).strip(), ["group_summary_table"]


def _build_group_metrics(raw_metrics: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_metrics, list) or not raw_metrics:
        return [{"op": "count", "as": "row_count"}]

    metrics: list[dict[str, Any]] = []
    for index, metric in enumerate(raw_metrics):
        if not isinstance(metric, dict):
            continue
        column = metric.get("column")
        agg = str(metric.get("agg") or "").strip().lower()
        semantic = str(metric.get("semantic") or "").strip().lower()
        if agg == "count":
            metrics.append({"op": "count", "as": "row_count"})
            continue
        if not isinstance(column, str) or not column.strip():
            continue
        op = "rate" if semantic == "rate" else agg
        if op not in {"mean", "median", "sum", "min", "max", "nunique", "rate"}:
            continue
        suffix = "rate" if op == "rate" else op
        metrics.append(
            {
                "column": column.strip(),
                "op": op,
                "as": f"{column.strip()}_{suffix}",
            }
        )
    return metrics or [{"op": "count", "as": "row_count"}]


def _render_markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "未生成可展示的分组统计结果。"

    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = [_format_cell(row.get(column)) for column in columns]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *body])


def _format_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)
