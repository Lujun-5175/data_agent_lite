from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data_manager import get_dataframe, get_model_prep_plan, get_schema_profile
from src.preprocessing import prepare_model_inputs
from src.tools import MLHelperAPI, SafeExecutionError


def _upload_df(client: TestClient, df: pd.DataFrame, filename: str = "ml.csv") -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    payload = buffer.getvalue().encode("utf-8")
    response = client.post("/upload", files={"file": (filename, payload, "text/csv")})
    assert response.status_code == 200
    return str(response.json()["dataset_id"])


def _classification_df() -> pd.DataFrame:
    rows = []
    for idx in range(120):
        monthly = 40 + (idx % 10) * 5
        tenure = 2 + (idx % 24)
        contract = "Month-to-month" if idx % 3 == 0 else "One year"
        churn = "Yes" if monthly > 65 or tenure < 8 else "No"
        rows.append(
            {
                "customer_id": f"c{idx:04d}",
                "tenure": tenure,
                "MonthlyCharges": float(monthly),
                "Contract": contract,
                "signup_date": f"2024-01-{(idx % 28) + 1:02d}",
                "notes": f"free text note {idx}",
                "Churn": churn,
            }
        )
    return pd.DataFrame(rows)


def _regression_df() -> pd.DataFrame:
    rows = []
    for idx in range(140):
        tenure = 1 + (idx % 30)
        monthly = 20 + (idx % 20) * 3
        contract = "monthly" if idx % 2 == 0 else "yearly"
        total = monthly * tenure * 0.95 + (8 if contract == "yearly" else 0)
        rows.append(
            {
                "cust_id": f"u{idx:04d}",
                "tenure": tenure,
                "MonthlyCharges": float(monthly),
                "Contract": contract,
                "TotalCharges": float(total),
            }
        )
    return pd.DataFrame(rows)


def test_logistic_fit_success(client: TestClient):
    dataset_id = _upload_df(client, _classification_df(), "classification.csv")
    df = get_dataframe(dataset_id)
    ml = MLHelperAPI(df, dataset_id=dataset_id)

    result = ml.logistic_fit(target="Churn")
    assert result["artifact_type"] == "model_result"
    assert result["model_type"] == "logistic_regression"
    assert result["target"] == "Churn"
    assert "metrics" in result and "accuracy" in result["metrics"]
    assert result["features_used"]
    excluded_cols = {item["column"] for item in result["excluded_columns"]}
    assert "customer_id" in excluded_cols
    assert "notes" in excluded_cols


def test_linear_regression_fit_success(client: TestClient):
    dataset_id = _upload_df(client, _regression_df(), "regression.csv")
    df = get_dataframe(dataset_id)
    ml = MLHelperAPI(df, dataset_id=dataset_id)

    result = ml.linear_regression_fit(target="TotalCharges")
    assert result["artifact_type"] == "model_result"
    assert result["model_type"] == "linear_regression"
    assert "metrics" in result
    assert set(result["metrics"].keys()) == {"rmse", "mae", "r2"}


def test_invalid_target_fails_cleanly(client: TestClient):
    dataset_id = _upload_df(client, _classification_df(), "invalid_target.csv")
    df = get_dataframe(dataset_id)
    ml = MLHelperAPI(df, dataset_id=dataset_id)
    with pytest.raises(SafeExecutionError):
        ml.logistic_fit(target="MissingTarget")


def test_explicit_positive_label_usage(client: TestClient):
    dataset_id = _upload_df(client, _classification_df(), "positive_label.csv")
    df = get_dataframe(dataset_id)
    ml = MLHelperAPI(df, dataset_id=dataset_id)
    result = ml.logistic_fit(target="Churn", positive_label="Yes")
    assert result["positive_label"] == "Yes"
    assert result["positive_label_source"] == "explicit"


def test_ambiguous_target_handling(client: TestClient):
    df = _classification_df()
    df["multi"] = [f"class_{i % 3}" for i in range(len(df.index))]
    dataset_id = _upload_df(client, df, "ambiguous_target.csv")
    ml = MLHelperAPI(get_dataframe(dataset_id), dataset_id=dataset_id)
    with pytest.raises(SafeExecutionError):
        ml.logistic_fit(target="multi")


def test_metrics_and_feature_importance_retrieval(client: TestClient):
    dataset_id = _upload_df(client, _classification_df(), "metrics.csv")
    ml = MLHelperAPI(get_dataframe(dataset_id), dataset_id=dataset_id)
    model = ml.logistic_fit(target="Churn")

    metrics = ml.metrics()
    assert metrics["artifact_type"] == "metrics_result"
    assert metrics["source_model_artifact_id"] == model["artifact_id"]
    assert "accuracy" in metrics["metrics"]

    importance = ml.feature_importance(top_k=5)
    assert importance["artifact_type"] == "feature_importance_result"
    assert importance["source_model_artifact_id"] == model["artifact_id"]
    assert len(importance["items"]) <= 5


def test_feature_and_row_bounds_enforced(client: TestClient):
    rows = 10050
    data: dict[str, list[float | int | str]] = {
        "target": [1 if idx % 2 == 0 else 0 for idx in range(rows)],
    }
    for col in range(40):
        data[f"f_{col:02d}"] = [float((idx + col) % 7) for idx in range(rows)]
    df = pd.DataFrame(data)
    dataset_id = _upload_df(client, df, "bounds.csv")
    ml = MLHelperAPI(get_dataframe(dataset_id), dataset_id=dataset_id)
    result = ml.logistic_fit(target="target")
    assert len(result["features_used"]) <= 30
    assert result["prep_summary"]["rows_used"] <= 10000
    assert result["warnings"]


def test_categorical_missing_values_become_unknown_not_nan(client: TestClient):
    df = _classification_df()
    df.loc[0, "Contract"] = None
    dataset_id = _upload_df(client, df, "cat_missing.csv")

    source_df = get_dataframe(dataset_id)
    profile = get_schema_profile(dataset_id)
    plan = get_model_prep_plan(dataset_id, target="Churn")
    bundle = prepare_model_inputs(source_df, profile, plan, target="Churn")

    contract_values = set(bundle.x["Contract"].astype(str).unique().tolist())
    assert "Unknown" in contract_values
    assert "nan" not in contract_values


def test_logistic_split_fails_gracefully_on_tiny_dataset(client: TestClient):
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature_b": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    dataset_id = _upload_df(client, df, "tiny_split.csv")
    ml = MLHelperAPI(get_dataframe(dataset_id), dataset_id=dataset_id)
    with pytest.raises(SafeExecutionError, match="样本量过小"):
        ml.logistic_fit(target="target")


def test_logistic_split_fails_gracefully_on_imbalanced_dataset(client: TestClient):
    df = pd.DataFrame(
        {
            "feature_a": list(range(20)),
            "feature_b": ["a" if i % 2 == 0 else "b" for i in range(20)],
            "target": [0] * 19 + [1],
        }
    )
    dataset_id = _upload_df(client, df, "imbalanced_split.csv")
    ml = MLHelperAPI(get_dataframe(dataset_id), dataset_id=dataset_id)
    with pytest.raises(SafeExecutionError, match="少数类样本不足|类别分布不满足"):
        ml.logistic_fit(target="target")
