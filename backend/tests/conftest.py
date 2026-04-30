from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import server, tools
from src.data_manager import cleanup_dataset_artifacts, dataset_store
from src.server import app


def _cleanup_all_datasets() -> None:
    for dataset in dataset_store.list_datasets():
        try:
            cleanup_dataset_artifacts(dataset.dataset_id)
        except Exception:
            continue


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    temp_data_dir = tmp_path / "temp_data"
    images_dir = tmp_path / "images"
    temp_data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    _cleanup_all_datasets()
    monkeypatch.setattr(server, "TEMP_DATA_DIR", temp_data_dir)
    monkeypatch.setattr(server, "IMAGES_DIR", images_dir)
    tools.EXECUTOR.image_dir = images_dir
    yield
    _cleanup_all_datasets()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
