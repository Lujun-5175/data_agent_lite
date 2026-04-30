from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import ModelPrepPlanError, plan_model_preprocessing, prepare_analysis_dataframe
from src.schema_profile import profile_dataframe


def _fixture_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"u{i}" for i in range(1, 9)],
            "tenure": [1, 2, 3, 4, 5, 6, 7, 8],
            "monthly_charges": [10.1, 20.2, 30.3, 40.4, 50.5, 60.6, 70.7, 80.8],
            "contract": ["M", "M", "Y", "Y", "M", "Y", "M", "Y"],
            "churn": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
            "signup_date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
                "2024-01-07",
                "2024-01-08",
            ],
            "notes": [
                "A long customer support note",
                "Another long customer support note",
                "Long text message for customer",
                "Long text message for customer again",
                "Free form text sample content",
                "Free form text sample 2",
                "Free form text sample 3",
                "Free form text sample 4",
            ],
        }
    )


def test_profile_dataframe_semantic_detection():
    profile = profile_dataframe(_fixture_df())
    columns = {item["column_name"]: item for item in profile["columns"]}

    assert columns["tenure"]["semantic_type"] == "numeric"
    assert columns["contract"]["semantic_type"] in {"categorical", "binary_label_candidate"}
    assert columns["signup_date"]["semantic_type"] == "datetime_like"
    assert columns["customer_id"]["semantic_type"] == "identifier_like"
    assert columns["churn"]["semantic_type"] == "binary_label_candidate"
    assert columns["notes"]["semantic_type"] == "text_like"
    assert columns["customer_id"]["usable_for_ml_feature"] is False


def test_profile_dataframe_missing_and_uniqueness_metadata():
    df = _fixture_df()
    df.loc[0, "contract"] = None
    profile = profile_dataframe(df)
    columns = {item["column_name"]: item for item in profile["columns"]}
    contract = columns["contract"]
    assert contract["missing_count"] == 1
    assert 0 < contract["missing_ratio"] < 1
    assert contract["unique_count"] >= 2
    assert isinstance(contract["sample_values"], list)


def test_prepare_analysis_dataframe_keeps_raw_unchanged_and_records_steps():
    raw = _fixture_df()
    raw_before = raw.copy(deep=True)
    profile = profile_dataframe(raw)
    analysis_df, artifact_payload, warnings = prepare_analysis_dataframe(raw, profile)

    pd.testing.assert_frame_equal(raw, raw_before)
    assert len(analysis_df.index) == len(raw.index)
    assert artifact_payload["stage"] == "analysis"
    assert artifact_payload["steps"]
    assert isinstance(warnings, list)


def test_model_prep_plan_explicit_target_and_feature_exclusion():
    df = _fixture_df()
    profile = profile_dataframe(df)
    payload, warnings = plan_model_preprocessing(df, profile, target="churn")

    assert payload["target"] == "churn"
    assert payload["target_status"] == "explicit_validated"
    excluded_columns = {entry["column"] for entry in payload["excluded_columns"]}
    assert "customer_id" in excluded_columns
    assert "notes" in excluded_columns
    assert "churn" not in payload["candidate_features"]
    assert isinstance(warnings, list)


def test_model_prep_plan_ambiguous_target_warning():
    df = _fixture_df()
    profile = profile_dataframe(df)
    payload, warnings = plan_model_preprocessing(df, profile)
    assert payload["target_status"] in {"ambiguous", "inferred_single", "not_found"}
    if payload["target_status"] == "ambiguous":
        assert warnings


def test_model_prep_plan_invalid_target_raises():
    df = _fixture_df()
    profile = profile_dataframe(df)
    with pytest.raises(ModelPrepPlanError):
        plan_model_preprocessing(df, profile, target="missing_target")
