from __future__ import annotations

import pandas as pd
import pytest

from src.multitable import (
    TableRelationship,
    build_join_plan,
    create_relationship_from_dict,
    estimate_join_risk,
    execute_join_plan,
    suggest_relationships_by_name,
    validate_relationship,
)


def _build_tables():
    left = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "sales": [10, 20],
        }
    )
    right = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "region": ["east", "west"],
        }
    )
    return {"left": left, "right": right}


def test_validate_relationship_passes_for_valid_keys():
    tables = _build_tables()
    relationship = TableRelationship(
        left_table="left",
        right_table="right",
        left_key="customer_id",
        right_key="customer_id",
        relationship_type="one_to_one",
    )

    warnings = validate_relationship(relationship, tables)

    assert isinstance(warnings, list)


def test_validate_relationship_raises_for_missing_table():
    tables = _build_tables()
    relationship = TableRelationship(
        left_table="left",
        right_table="missing",
        left_key="customer_id",
        right_key="customer_id",
        relationship_type="one_to_one",
    )

    with pytest.raises(ValueError, match="Missing table"):
        validate_relationship(relationship, tables)


def test_validate_relationship_raises_for_missing_key():
    tables = _build_tables()
    relationship = TableRelationship(
        left_table="left",
        right_table="right",
        left_key="missing_key",
        right_key="customer_id",
        relationship_type="one_to_one",
    )

    with pytest.raises(ValueError, match="Missing key column"):
        validate_relationship(relationship, tables)


def test_estimate_join_risk_detects_many_to_many():
    left = pd.DataFrame({"customer_id": [1, 1, 2], "sales": [10, 11, 20]})
    right = pd.DataFrame({"customer_id": [1, 1, 3], "region": ["east", "west", "north"]})

    risk = estimate_join_risk(
        left,
        right,
        left_key="customer_id",
        right_key="customer_id",
        relationship_type="many_to_many",
    )

    assert risk["likely_many_to_many"] is True
    assert risk["warnings"]


def test_build_join_plan_requires_approval_for_inferred_key():
    tables = _build_tables()

    plan = build_join_plan(
        left_table="left",
        right_table="right",
        left_key="customer_id",
        right_key="customer_id",
        join_type="left",
        relationship_type="one_to_one",
        tables=tables,
        inferred_key=True,
    )

    assert plan.requires_approval is True
    assert any(risk.risk_type == "inferred_join_key" for risk in plan.risks)


def test_build_join_plan_warns_for_non_unique_right_key_many_to_one():
    left = pd.DataFrame({"customer_id": [1, 2, 3], "sales": [10, 20, 30]})
    right = pd.DataFrame({"customer_id": [1, 1, 2], "region": ["east", "west", "north"]})
    tables = {"left": left, "right": right}

    plan = build_join_plan(
        left_table="left",
        right_table="right",
        left_key="customer_id",
        right_key="customer_id",
        join_type="left",
        relationship_type="many_to_one",
        tables=tables,
    )

    assert any(risk.risk_type == "relationship_mismatch" for risk in plan.risks)
    assert plan.requires_approval is True


def test_execute_join_plan_requires_approval_before_execution():
    tables = _build_tables()
    plan = build_join_plan(
        left_table="left",
        right_table="right",
        left_key="customer_id",
        right_key="customer_id",
        join_type="left",
        relationship_type="one_to_one",
        tables=tables,
        inferred_key=True,
    )

    joined_df, result = execute_join_plan(plan, tables, approved=False)

    assert joined_df is None
    assert result.status == "requires_approval"
    assert result.approval_request is not None
    assert result.approval_request["approval_id"]


def test_execute_join_plan_runs_after_approval():
    tables = _build_tables()
    plan = build_join_plan(
        left_table="left",
        right_table="right",
        left_key="customer_id",
        right_key="customer_id",
        join_type="left",
        relationship_type="one_to_one",
        tables=tables,
        inferred_key=True,
    )

    joined_df, result = execute_join_plan(plan, tables, approved=True)

    assert joined_df is not None
    assert result.status == "success"
    assert result.joined_rows == 2
    assert list(joined_df.columns) == ["customer_id", "sales", "region"]


def test_execute_join_plan_safe_without_approval_for_declared_low_risk_relationship():
    tables = _build_tables()
    plan = build_join_plan(
        left_table="left",
        right_table="right",
        left_key="customer_id",
        right_key="customer_id",
        join_type="left",
        relationship_type="one_to_one",
        tables=tables,
    )

    joined_df, result = execute_join_plan(plan, tables, approved=False)

    assert plan.requires_approval is False
    assert joined_df is not None
    assert result.status == "success"
    assert result.joined_rows == 2


def test_join_does_not_mutate_original_dataframes():
    tables = _build_tables()
    original_left = tables["left"].copy(deep=True)
    original_right = tables["right"].copy(deep=True)
    plan = build_join_plan(
        left_table="left",
        right_table="right",
        left_key="customer_id",
        right_key="customer_id",
        join_type="left",
        relationship_type="one_to_one",
        tables=tables,
    )

    execute_join_plan(plan, tables, approved=False)

    pd.testing.assert_frame_equal(tables["left"], original_left)
    pd.testing.assert_frame_equal(tables["right"], original_right)


def test_suggest_relationships_by_name_finds_customer_id_if_implemented():
    tables = _build_tables()
    suggestions = suggest_relationships_by_name(tables)

    assert any(
        rel.left_key == "customer_id"
        and rel.right_key == "customer_id"
        and rel.declared is False
        for rel in suggestions
    )


def test_create_relationship_from_dict_roundtrip():
    relationship = create_relationship_from_dict(
        {
            "left_table": "left",
            "right_table": "right",
            "left_key": "customer_id",
            "right_key": "customer_id",
            "relationship_type": "one_to_one",
            "declared": True,
            "confidence": 0.9,
            "notes": "test",
        }
    )

    assert relationship.left_table == "left"
    assert relationship.right_table == "right"
    assert relationship.declared is True
