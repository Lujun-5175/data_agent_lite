from __future__ import annotations

from fastapi.testclient import TestClient

from src.data_manager import get_analysis_preprocess_artifact, get_dataset, get_dataframe, get_model_prep_plan, get_schema_profile


def test_preprocessing_log_and_working_df_semantics(client: TestClient):
    csv_content = b"num,cat\n1,a\n, \n2,\n,\n"
    upload_response = client.post(
        "/upload",
        files={"file": ("dirty.csv", csv_content, "text/csv")},
    )
    assert upload_response.status_code == 200
    payload = upload_response.json()
    dataset_id = payload["dataset_id"]

    # Upload path should be minimal and explicit about deferred preprocessing.
    assert payload["preprocessing_log"]
    assert "上传阶段仅完成基础读取" in payload["preprocessing_log"][0]

    dataset_before = get_dataset(dataset_id)
    assert dataset_before.preprocessed is False
    assert dataset_before.analysis_basis == "raw_df"
    assert dataset_before.schema_profile_artifact["artifact_type"] == "schema_profile"
    assert dataset_before.analysis_preprocess_artifact is None
    schema = get_schema_profile(dataset_id)
    assert schema["artifact_type"] == "schema_profile"
    assert schema["dataset_id"] == dataset_id

    original_null_count = int(dataset_before.original_df.isna().sum().sum())
    _ = get_dataframe(dataset_id)
    dataset_after = get_dataset(dataset_id)
    working_null_count = int(dataset_after.working_df.isna().sum().sum())
    preprocess_artifact = get_analysis_preprocess_artifact(dataset_id)

    assert dataset_after.preprocessed is True
    assert dataset_after.analysis_basis == "analysis_df"
    assert dataset_after.preprocessing_log
    assert any(
        any(keyword in entry for keyword in ("coerce", "drop_all_null", "noop", "warning"))
        for entry in dataset_after.preprocessing_log
    )
    assert preprocess_artifact["artifact_type"] == "preprocess_result"
    assert int(dataset_after.original_df.isna().sum().sum()) == original_null_count
    assert working_null_count <= original_null_count + 1


def test_model_prep_plan_generated_on_demand(client: TestClient):
    csv_content = b"id,churn,fee,note\nu1,Yes,11.0,hello world\nu2,No,9.0,long text example\n"
    upload_response = client.post(
        "/upload",
        files={"file": ("prep.csv", csv_content, "text/csv")},
    )
    assert upload_response.status_code == 200
    dataset_id = upload_response.json()["dataset_id"]

    dataset = get_dataset(dataset_id)
    assert dataset.model_prep_plan_artifact is None

    plan = get_model_prep_plan(dataset_id, target="churn")
    assert plan["artifact_type"] == "model_prep_plan"
    assert plan["target"] == "churn"
    assert plan["target_status"] == "explicit_validated"

    dataset_after = get_dataset(dataset_id)
    assert dataset_after.model_prep_plan_artifact is not None
