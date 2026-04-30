from __future__ import annotations

import pandas as pd
import pytest

from src.tools import SafeExecutionError, StatsHelperAPI


@pytest.fixture
def stats_helper():
    df = pd.DataFrame(
        {
            "segment": ["A", "A", "B", "B", "C", "C"],
            "country": ["US", "US", "US", "CA", "CA", "US"],
            "sales": [100, 120, 80, 90, 60, 70],
            "tenure": [1, 2, 2, 3, 3, 4],
            "churn": ["yes", "no", "yes", "no", "no", "yes"],
        }
    )
    return StatsHelperAPI(df, dataset_id="test-dataset")


def test_describe_numeric(stats_helper: StatsHelperAPI):
    artifact = stats_helper.describe_numeric(["sales", "tenure"])
    assert artifact["artifact_type"] == "stats_result"
    assert artifact["stats_type"] == "describe_numeric"
    assert artifact["columns"] == ["sales", "tenure"]
    first = artifact["rows"][0]
    assert set(["count", "mean", "std", "min", "25%", "50%", "75%", "max", "missing_count"]).issubset(first.keys())


def test_describe_categorical(stats_helper: StatsHelperAPI):
    artifact = stats_helper.describe_categorical(["segment", "country"])
    assert artifact["artifact_type"] == "stats_result"
    assert artifact["stats_type"] == "describe_categorical"
    assert artifact["rows"][0]["unique_count"] >= 2


def test_group_summary_and_sort_top_n(stats_helper: StatsHelperAPI):
    artifact = stats_helper.group_summary(
        group_by="country",
        metrics=[{"op": "mean", "column": "sales", "as": "avg_sales"}, {"op": "count", "as": "rows"}],
        sort_by="avg_sales",
        ascending=False,
        top_n=1,
    )
    assert artifact["artifact_type"] == "stats_result"
    assert artifact["stats_type"] == "group_summary"
    assert artifact["row_count"] == 1
    assert artifact["rows"][0]["group"] == "US"


def test_rate_metric_with_explicit_positive_label(stats_helper: StatsHelperAPI):
    artifact = stats_helper.group_summary(
        group_by="country",
        metrics=[{"op": "rate", "column": "churn", "as": "churn_rate", "positive_label": "yes"}],
        sort_by="group",
        ascending=True,
        top_n=10,
    )
    assert artifact["artifact_type"] == "stats_result"
    assert artifact["rate_metadata"][0]["positive_label"] == "yes"
    assert artifact["rate_metadata"][0]["positive_label_source"] == "explicit"


def test_rate_metric_binary_numeric_default():
    df = pd.DataFrame({"group": ["A", "A", "B", "B"], "target": [1, 0, 1, 1]})
    helper = StatsHelperAPI(df, dataset_id="test-dataset")
    artifact = helper.group_summary(
        group_by="group",
        metrics=[{"op": "rate", "column": "target", "as": "target_rate"}],
        sort_by="group",
        ascending=True,
    )
    metadata = artifact["rate_metadata"][0]
    assert metadata["positive_label"] == 1
    assert metadata["positive_label_source"] == "binary_numeric_default"


def test_rate_metric_ambiguous_string_warns():
    df = pd.DataFrame({"group": ["A", "A", "B", "B"], "label": ["Alpha", "Beta", "Alpha", "Beta"]})
    helper = StatsHelperAPI(df, dataset_id="test-dataset")
    artifact = helper.group_summary(
        group_by="group",
        metrics=[{"op": "rate", "column": "label", "as": "label_rate"}],
        sort_by="group",
        ascending=True,
    )
    metadata = artifact["rate_metadata"][0]
    assert metadata["positive_label_source"] == "ambiguous"
    assert metadata["positive_label_warning"]
    assert artifact["warnings"]


def test_correlation_shape(stats_helper: StatsHelperAPI):
    artifact = stats_helper.correlation(["sales", "tenure"], top_k=5)
    assert artifact["artifact_type"] == "stats_result"
    assert artifact["stats_type"] == "correlation"
    assert len(artifact["matrix"]) == 2
    assert len(artifact["top_pairs"]) == 1


def test_t_test_structure(stats_helper: StatsHelperAPI):
    artifact = stats_helper.t_test("sales", "country", "US", "CA")
    assert artifact["artifact_type"] == "test_result"
    assert artifact["test_type"] == "t_test"
    assert "p_value" in artifact
    assert artifact["group_a_size"] > 0


def test_chi_square_structure(stats_helper: StatsHelperAPI):
    artifact = stats_helper.chi_square("country", "churn")
    assert artifact["artifact_type"] == "test_result"
    assert artifact["test_type"] == "chi_square"
    assert "dof" in artifact


def test_anova_structure(stats_helper: StatsHelperAPI):
    artifact = stats_helper.anova("sales", "segment")
    assert artifact["artifact_type"] == "test_result"
    assert artifact["test_type"] == "anova"
    assert len(artifact["group_stats"]) >= 3


def test_invalid_column_fails_cleanly(stats_helper: StatsHelperAPI):
    with pytest.raises(SafeExecutionError):
        stats_helper.describe_numeric(["not_exist"])


def test_invalid_type_combo_fails_cleanly(stats_helper: StatsHelperAPI):
    with pytest.raises(SafeExecutionError):
        stats_helper.t_test("country", "segment", "A", "B")
