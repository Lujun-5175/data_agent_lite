from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
backend_path = str(BACKEND_DIR)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from src import chat_service, intent_planner, server  # noqa: E402
from src.data_manager import cleanup_dataset_artifacts, dataset_store  # noqa: E402
from src.server import app  # noqa: E402


def _cleanup_all_datasets() -> None:
    for dataset in dataset_store.list_datasets():
        try:
            cleanup_dataset_artifacts(dataset.dataset_id)
        except Exception:
            continue


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    temp_data_dir = tmp_path / "temp_data"
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    _cleanup_all_datasets()
    monkeypatch.setattr(server, "TEMP_DATA_DIR", temp_data_dir)
    monkeypatch.setattr(intent_planner, "INTENT_PLANNER_MODEL", None)
    monkeypatch.setattr(intent_planner, "get_intent_planner_model", lambda: intent_planner.INTENT_PLANNER_MODEL)
    monkeypatch.setattr(chat_service, "get_intent_planner_model", lambda: intent_planner.INTENT_PLANNER_MODEL)
    yield
    _cleanup_all_datasets()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
