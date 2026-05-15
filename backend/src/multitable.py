from __future__ import annotations

from typing import Any
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.approval import (
    ApprovalRisk,
    create_approval_request,
    requires_approval as approval_requires_approval,
    serialize_approval_request,
    detect_risky_operation,
)
from src.settings import SETTINGS


class TableRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_table: str
    right_table: str
    left_key: str
    right_key: str
    relationship_type: str
    declared: bool = True
    confidence: float | None = None
    notes: str | None = None


class JoinPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    join_type: str
    relationship_type: str
    inferred_key: bool
    requires_approval: bool
    risks: list[ApprovalRisk] = Field(default_factory=list)
    estimated_left_rows: int | None = None
    estimated_right_rows: int | None = None
    estimated_output_rows: int | None = None
    unmatched_left_rate: float | None = None
    unmatched_right_rate: float | None = None
    metadata: dict[str, Any] | None = None


class JoinResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    status: str
    joined_rows: int | None = None
    joined_columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    approval_request: dict[str, Any] | None = None
    dataframe_preview: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None


_JOIN_TYPES = {"inner", "left", "right", "outer"}
_RELATIONSHIP_TYPES = {"one_to_one", "one_to_many", "many_to_one", "many_to_many", "unknown"}
_HIGH_NULL_RATE = 0.3
_UNIQUE_THRESHOLD = 0.98


def validate_relationship(
    relationship: TableRelationship,
    tables: dict[str, pd.DataFrame],
) -> list[str]:
    left_df = _require_table(tables, relationship.left_table)
    right_df = _require_table(tables, relationship.right_table)
    _require_key(left_df, relationship.left_key, relationship.left_table)
    _require_key(right_df, relationship.right_key, relationship.right_table)

    warnings: list[str] = []
    left_series = left_df[relationship.left_key]
    right_series = right_df[relationship.right_key]

    if not _joinable_dtype_pair(left_series.dtype, right_series.dtype):
        warnings.append(
            f"Join key dtypes may be incompatible: {relationship.left_table}.{relationship.left_key} "
            f"({left_series.dtype}) vs {relationship.right_table}.{relationship.right_key} ({right_series.dtype})."
        )

    left_null_rate = _null_rate(left_series)
    right_null_rate = _null_rate(right_series)
    if left_null_rate is not None and left_null_rate > _HIGH_NULL_RATE:
        warnings.append(
            f"{relationship.left_table}.{relationship.left_key} has a high null rate ({left_null_rate:.2%})."
        )
    if right_null_rate is not None and right_null_rate > _HIGH_NULL_RATE:
        warnings.append(
            f"{relationship.right_table}.{relationship.right_key} has a high null rate ({right_null_rate:.2%})."
        )

    left_unique_ratio = _unique_ratio(left_series)
    right_unique_ratio = _unique_ratio(right_series)
    if relationship.relationship_type == "many_to_one" and right_unique_ratio < _UNIQUE_THRESHOLD:
        warnings.append(
            f"{relationship.right_table}.{relationship.right_key} is not unique, so many_to_one is unlikely to hold."
        )
    if relationship.relationship_type == "one_to_many" and left_unique_ratio < _UNIQUE_THRESHOLD:
        warnings.append(
            f"{relationship.left_table}.{relationship.left_key} is not unique, so one_to_many is unlikely to hold."
        )
    if relationship.relationship_type == "many_to_many":
        warnings.append("Declared many_to_many join may expand rows substantially.")
    if left_unique_ratio < _UNIQUE_THRESHOLD and right_unique_ratio < _UNIQUE_THRESHOLD:
        warnings.append("Both join keys contain duplicates; many_to_many behavior is likely.")

    return warnings


def estimate_join_risk(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    left_key: str,
    right_key: str,
    relationship_type: str,
) -> dict[str, Any]:
    left_series = left_df[left_key]
    right_series = right_df[right_key]

    left_rows = int(len(left_df.index))
    right_rows = int(len(right_df.index))
    left_key_null_rate = _null_rate(left_series)
    right_key_null_rate = _null_rate(right_series)
    left_unique_ratio = _unique_ratio(left_series)
    right_unique_ratio = _unique_ratio(right_series)
    sampled_overlap_rate = _overlap_rate(left_series, right_series)
    likely_many_to_many = (
        left_unique_ratio < _UNIQUE_THRESHOLD
        and right_unique_ratio < _UNIQUE_THRESHOLD
        and sampled_overlap_rate > 0.0
    )

    warnings: list[str] = []
    if left_key_null_rate is not None and left_key_null_rate > _HIGH_NULL_RATE:
        warnings.append(f"left key null rate is high ({left_key_null_rate:.2%}).")
    if right_key_null_rate is not None and right_key_null_rate > _HIGH_NULL_RATE:
        warnings.append(f"right key null rate is high ({right_key_null_rate:.2%}).")
    if likely_many_to_many:
        warnings.append("Join is likely many_to_many because both keys are non-unique and overlap.")
    if relationship_type == "many_to_one" and right_unique_ratio < _UNIQUE_THRESHOLD:
        warnings.append("Declared many_to_one but right key is not unique.")
    if relationship_type == "one_to_many" and left_unique_ratio < _UNIQUE_THRESHOLD:
        warnings.append("Declared one_to_many but left key is not unique.")

    estimated_output_rows = _estimate_output_rows(
        left_rows=left_rows,
        right_rows=right_rows,
        left_unique_ratio=left_unique_ratio,
        right_unique_ratio=right_unique_ratio,
        relationship_type=relationship_type,
        likely_many_to_many=likely_many_to_many,
        sampled_overlap_rate=sampled_overlap_rate,
    )

    unmatched_left_rate = _unmatched_rate(left_series, right_series)
    unmatched_right_rate = _unmatched_rate(right_series, left_series)

    return {
        "left_rows": left_rows,
        "right_rows": right_rows,
        "left_key_null_rate": left_key_null_rate,
        "right_key_null_rate": right_key_null_rate,
        "left_unique_ratio": left_unique_ratio,
        "right_unique_ratio": right_unique_ratio,
        "sampled_overlap_rate": sampled_overlap_rate,
        "likely_many_to_many": likely_many_to_many,
        "warnings": warnings,
        "estimated_output_rows": estimated_output_rows,
        "unmatched_left_rate": unmatched_left_rate,
        "unmatched_right_rate": unmatched_right_rate,
    }


def build_join_plan(
    *,
    left_table: str,
    right_table: str,
    left_key: str,
    right_key: str,
    join_type: str = "left",
    relationship_type: str = "unknown",
    tables: dict[str, pd.DataFrame],
    inferred_key: bool = False,
) -> JoinPlan:
    if join_type not in _JOIN_TYPES:
        raise ValueError(f"Unsupported join type: {join_type}")
    if relationship_type not in _RELATIONSHIP_TYPES:
        raise ValueError(f"Unsupported relationship type: {relationship_type}")

    relationship = TableRelationship(
        left_table=left_table,
        right_table=right_table,
        left_key=left_key,
        right_key=right_key,
        relationship_type=relationship_type,
        declared=True,
        confidence=None,
        notes=None,
    )
    validation_warnings = validate_relationship(relationship, tables)
    left_df = tables[left_table]
    right_df = tables[right_table]
    risk_estimate = estimate_join_risk(
        left_df,
        right_df,
        left_key=left_key,
        right_key=right_key,
        relationship_type=relationship_type,
    )

    risks = list(detect_risky_operation({"action": "join", "inferred_key": inferred_key}))
    if relationship_type == "unknown":
        risks.append(
            ApprovalRisk(
                risk_type="unknown_relationship",
                severity="medium",
                message="The relationship was not explicitly declared and should be reviewed.",
                metadata={
                    "left_table": left_table,
                    "right_table": right_table,
                    "left_key": left_key,
                    "right_key": right_key,
                },
            )
        )
    if risk_estimate["likely_many_to_many"]:
        risks.append(
            ApprovalRisk(
                risk_type="many_to_many_join",
                severity="high",
                message="The join is likely many-to-many and may expand rows.",
                metadata={
                    "left_table": left_table,
                    "right_table": right_table,
                    "left_key": left_key,
                    "right_key": right_key,
                },
            )
        )
    if relationship_type == "one_to_one" and (
        (risk_estimate["left_unique_ratio"] is not None and risk_estimate["left_unique_ratio"] < _UNIQUE_THRESHOLD)
        or (risk_estimate["right_unique_ratio"] is not None and risk_estimate["right_unique_ratio"] < _UNIQUE_THRESHOLD)
    ):
        risks.append(
            ApprovalRisk(
                risk_type="relationship_mismatch",
                severity="high",
                message="Declared one_to_one relationship is not supported by key uniqueness.",
                metadata={
                    "left_table": left_table,
                    "right_table": right_table,
                    "left_key": left_key,
                    "right_key": right_key,
                },
            )
        )
    if relationship_type == "many_to_many":
        risks.append(
            ApprovalRisk(
                risk_type="many_to_many_join",
                severity="high",
                message="Declared many_to_many joins should be reviewed before execution.",
                metadata={
                    "left_table": left_table,
                    "right_table": right_table,
                    "left_key": left_key,
                    "right_key": right_key,
                },
            )
        )
    if relationship_type == "many_to_one" and risk_estimate["right_unique_ratio"] is not None and risk_estimate["right_unique_ratio"] < _UNIQUE_THRESHOLD:
        risks.append(
            ApprovalRisk(
                risk_type="relationship_mismatch",
                severity="high",
                message="Declared many_to_one relationship is not supported by right key uniqueness.",
                metadata={
                    "left_table": left_table,
                    "right_table": right_table,
                    "left_key": left_key,
                    "right_key": right_key,
                },
            )
        )
    if relationship_type == "one_to_many" and risk_estimate["left_unique_ratio"] is not None and risk_estimate["left_unique_ratio"] < _UNIQUE_THRESHOLD:
        risks.append(
            ApprovalRisk(
                risk_type="relationship_mismatch",
                severity="high",
                message="Declared one_to_many relationship is not supported by left key uniqueness.",
                metadata={
                    "left_table": left_table,
                    "right_table": right_table,
                    "left_key": left_key,
                    "right_key": right_key,
                },
            )
        )

    metadata = {
        "validation_warnings": validation_warnings,
        "risk_estimate": risk_estimate,
    }
    requires_join_approval = approval_requires_approval(risks)
    return JoinPlan(
        plan_id=uuid4().hex,
        left_table=left_table,
        right_table=right_table,
        left_key=left_key,
        right_key=right_key,
        join_type=join_type,
        relationship_type=relationship_type,
        inferred_key=inferred_key,
        requires_approval=requires_join_approval,
        risks=risks,
        estimated_left_rows=risk_estimate["left_rows"],
        estimated_right_rows=risk_estimate["right_rows"],
        estimated_output_rows=risk_estimate["estimated_output_rows"],
        unmatched_left_rate=risk_estimate["unmatched_left_rate"],
        unmatched_right_rate=risk_estimate["unmatched_right_rate"],
        metadata=metadata,
    )


def execute_join_plan(
    plan: JoinPlan,
    tables: dict[str, pd.DataFrame],
    *,
    approved: bool = False,
    preview_rows: int = 20,
) -> tuple[pd.DataFrame | None, JoinResult]:
    if not SETTINGS.multitable_enabled:
        return None, JoinResult(
            plan_id=plan.plan_id,
            status="error",
            warnings=["Multitable joins are disabled in settings."],
            error_message="Multitable joins are disabled in settings.",
        )

    if plan.requires_approval and not approved:
        approval_request = create_approval_request(
            action_type="multitable_join",
            proposed_action=f"{plan.left_table} {plan.join_type} join {plan.right_table} on {plan.left_key}={plan.right_key}",
            risks=plan.risks,
            metadata={
                "plan_id": plan.plan_id,
                "left_table": plan.left_table,
                "right_table": plan.right_table,
                "join_type": plan.join_type,
                "relationship_type": plan.relationship_type,
                "inferred_key": plan.inferred_key,
            },
        )
        return None, JoinResult(
            plan_id=plan.plan_id,
            status="requires_approval",
            warnings=["Join requires approval before execution."],
            approval_request=serialize_approval_request(approval_request),
        )

    try:
        left_df = tables[plan.left_table]
        right_df = tables[plan.right_table]
        merged = _merge_tables(plan, left_df, right_df)
        preview_limit = min(max(0, int(preview_rows)), SETTINGS.multitable_max_preview_rows)
        preview = []
        if preview_limit > 0:
            preview = _frame_to_records(merged.head(preview_limit))

        warnings = list(plan.metadata.get("validation_warnings", [])) if isinstance(plan.metadata, dict) else []
        warnings.extend(_join_runtime_warnings(merged, plan))
        result = JoinResult(
            plan_id=plan.plan_id,
            status="success",
            joined_rows=int(len(merged.index)),
            joined_columns=[str(column) for column in merged.columns],
            warnings=warnings,
            dataframe_preview=preview,
        )
        return merged, result
    except Exception as exc:
        return None, JoinResult(
            plan_id=plan.plan_id,
            status="error",
            warnings=[],
            error_message=str(exc),
        )


def create_relationship_from_dict(data: dict[str, Any]) -> TableRelationship:
    return TableRelationship.model_validate(
        {
            "left_table": data["left_table"],
            "right_table": data["right_table"],
            "left_key": data["left_key"],
            "right_key": data["right_key"],
            "relationship_type": data.get("relationship_type", "unknown"),
            "declared": bool(data.get("declared", True)),
            "confidence": data.get("confidence"),
            "notes": data.get("notes"),
        }
    )


def suggest_relationships_by_name(tables: dict[str, pd.DataFrame]) -> list[TableRelationship]:
    suggestions: list[TableRelationship] = []
    table_names = list(tables.keys())
    common_id_like = {"customer_id", "user_id", "order_id", "account_id", "product_id"}

    for i, left_name in enumerate(table_names):
        left_df = tables[left_name]
        for right_name in table_names[i + 1 :]:
            right_df = tables[right_name]
            shared_columns = sorted(set(str(col) for col in left_df.columns) & set(str(col) for col in right_df.columns))
            for column in shared_columns:
                if column.lower().endswith("_id") or column.lower() in common_id_like:
                    suggestions.append(
                        TableRelationship(
                            left_table=left_name,
                            right_table=right_name,
                            left_key=column,
                            right_key=column,
                            relationship_type="unknown",
                            declared=False,
                            confidence=0.75 if column.lower() in common_id_like else 0.6,
                            notes="Name-based suggestion only; approval is required before use.",
                        )
                    )
    return suggestions


def _require_table(tables: dict[str, pd.DataFrame], table_name: str) -> pd.DataFrame:
    if table_name not in tables:
        raise ValueError(f"Missing table: {table_name}")
    table = tables[table_name]
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"Table is not a DataFrame: {table_name}")
    return table


def _require_key(df: pd.DataFrame, key: str, table_name: str) -> None:
    if key not in df.columns:
        raise ValueError(f"Missing key column {key} in table {table_name}")


def _null_rate(series: pd.Series) -> float | None:
    if len(series.index) == 0:
        return None
    return float(series.isna().mean())


def _unique_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if len(non_null.index) == 0:
        return 0.0
    return float(non_null.nunique(dropna=True) / len(non_null.index))


def _overlap_rate(left_series: pd.Series, right_series: pd.Series) -> float:
    left_values = pd.unique(left_series.dropna())
    right_values = pd.unique(right_series.dropna())
    if len(left_values) == 0 or len(right_values) == 0:
        return 0.0
    shared = len(set(left_values.tolist()).intersection(set(right_values.tolist())))
    denominator = float(min(len(left_values), len(right_values)))
    if denominator <= 0:
        return 0.0
    return float(shared / denominator)


def _estimate_output_rows(
    *,
    left_rows: int,
    right_rows: int,
    left_unique_ratio: float,
    right_unique_ratio: float,
    relationship_type: str,
    likely_many_to_many: bool,
    sampled_overlap_rate: float,
) -> int:
    if relationship_type == "inner":
        baseline = min(left_rows, right_rows)
    elif relationship_type == "right":
        baseline = right_rows
    elif relationship_type == "outer":
        baseline = max(left_rows, right_rows)
    else:
        baseline = left_rows

    if likely_many_to_many:
        expansion_factor = max(1.0, (1.0 / max(left_unique_ratio, 0.1)) * (1.0 / max(right_unique_ratio, 0.1)))
        expansion_factor = min(expansion_factor, 10.0)
        return max(baseline, int(round(baseline * max(1.0, sampled_overlap_rate) * expansion_factor)))

    if relationship_type == "inner":
        return int(round(baseline * max(0.0, sampled_overlap_rate)))
    return int(round(baseline))


def _unmatched_rate(source_series: pd.Series, target_series: pd.Series) -> float | None:
    non_null = source_series.dropna()
    if len(non_null.index) == 0:
        return None
    target_values = set(pd.unique(target_series.dropna()).tolist())
    unmatched = (~non_null.isin(target_values)).mean()
    return float(unmatched)


def _joinable_dtype_pair(left_dtype: Any, right_dtype: Any) -> bool:
    left_family = _dtype_family(left_dtype)
    right_family = _dtype_family(right_dtype)
    if left_family == right_family:
        return True
    if {left_family, right_family} <= {"string", "categorical"}:
        return True
    return False


def _dtype_family(dtype: Any) -> str:
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return "string"
    return "other"


def _merge_tables(plan: JoinPlan, left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
    merge_kwargs: dict[str, Any] = {
        "how": plan.join_type,
        "suffixes": ("_left", "_right"),
    }
    if plan.left_key == plan.right_key:
        merge_kwargs["on"] = plan.left_key
    else:
        merge_kwargs["left_on"] = plan.left_key
        merge_kwargs["right_on"] = plan.right_key
    if plan.relationship_type in _RELATIONSHIP_TYPES - {"unknown"}:
        merge_kwargs["validate"] = plan.relationship_type
    return left_df.merge(right_df, **merge_kwargs)


def _join_runtime_warnings(joined: pd.DataFrame, plan: JoinPlan) -> list[str]:
    warnings: list[str] = []
    if plan.estimated_output_rows is not None and len(joined.index) > max(plan.estimated_output_rows, 0):
        warnings.append(
            f"Joined rows ({len(joined.index)}) exceeded the estimated output rows ({plan.estimated_output_rows})."
        )
    if len(joined.index) > max(plan.estimated_left_rows or 0, plan.estimated_right_rows or 0) * 2:
        warnings.append("Join expanded rows substantially.")
    return warnings


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = frame.copy(deep=True).astype(object)
    normalized = normalized.where(pd.notna(normalized), None)
    return normalized.to_dict(orient="records")
