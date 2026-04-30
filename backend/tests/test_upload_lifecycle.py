from __future__ import annotations

from fastapi.testclient import TestClient

from src import server
from src.data_manager import get_model_prep_plan
from src.result_types import artifact_registry


def test_reject_non_csv_upload(client: TestClient):
    response = client.post(
        "/upload",
        files={"file": ("bad.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "invalid_file_type"


def test_reject_file_over_size_limit(client: TestClient, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_SIZE_BYTES", 1024)
    too_large = b"a" * 2048
    response = client.post(
        "/upload",
        files={"file": ("large.csv", too_large, "text/csv")},
    )
    assert response.status_code == 413
    payload = response.json()
    assert payload["error"]["code"] == "file_too_large"


def test_upload_success_and_delete_lifecycle(client: TestClient):
    csv_content = b"num,cat\n1,a\n2,b\n3,c\n"
    upload_response = client.post(
        "/upload",
        files={"file": ("sample.csv", csv_content, "text/csv")},
    )
    assert upload_response.status_code == 200
    upload_payload = upload_response.json()
    assert upload_payload["dataset_id"]
    assert isinstance(upload_payload["preview"], list)
    assert upload_payload["analysis_basis"] == "raw_df"

    dataset_id = upload_payload["dataset_id"]
    _ = get_model_prep_plan(dataset_id)
    assert artifact_registry.get_latest(dataset_id) is not None

    delete_response = client.delete(f"/datasets/{dataset_id}")
    assert delete_response.status_code == 200
    assert artifact_registry.get_latest(dataset_id) is None

    after_delete = client.get("/data-preview", params={"dataset_id": dataset_id})
    assert after_delete.status_code == 404
    assert after_delete.json()["error"]["code"] == "dataset_not_found"
