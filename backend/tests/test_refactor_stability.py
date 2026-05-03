from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse

from src import data_manager, server
from src.tools import get_current_dataset_id


def _upload_dataset(client: TestClient, name: str = "sample.csv") -> str:
    response = client.post(
        "/upload",
        files={"file": (name, b"num,cat\n1,a\n2,b\n3,c\n", "text/csv")},
    )
    assert response.status_code == 200
    return str(response.json()["dataset_id"])


class _ContextEchoGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        dataset_id = config.get("configurable", {}).get("dataset_id")
        assert get_current_dataset_id() == dataset_id
        await asyncio.sleep(0.05)
        yield {
            "event": "on_chat_model_stream",
            "name": "echo",
            "data": {"chunk": f"dataset={context.dataset_id} current={get_current_dataset_id()}"},
        }


class _FailingContextGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        assert get_current_dataset_id() == context.dataset_id
        raise RuntimeError("boom")
        yield  # pragma: no cover


def test_chat_stream_clears_dataset_context_after_success(client: TestClient, monkeypatch):
    dataset_id = _upload_dataset(client)
    monkeypatch.setattr(server, "graph", _ContextEchoGraph())

    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": "分析一下"}]},
        },
    )

    assert response.status_code == 200
    assert f"dataset={dataset_id}" in response.text
    assert get_current_dataset_id() is None


def test_chat_stream_clears_dataset_context_after_graph_failure(client: TestClient, monkeypatch):
    dataset_id = _upload_dataset(client)
    monkeypatch.setattr(server, "graph", _FailingContextGraph())

    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": [{"type": "human", "content": "分析一下"}]},
        },
    )

    assert response.status_code == 200
    assert "internal_error" in response.text
    assert get_current_dataset_id() is None


def test_concurrent_chat_stream_requests_do_not_mix_dataset_contexts(client: TestClient, monkeypatch):
    dataset_id_1 = _upload_dataset(client, "left.csv")
    dataset_id_2 = _upload_dataset(client, "right.csv")
    monkeypatch.setattr(server, "graph", _ContextEchoGraph())

    responses: dict[str, str] = {}

    def _run(dataset_id: str):
        with TestClient(server.app) as threaded_client:
            response = threaded_client.post(
                "/chat/stream",
                json={
                    "dataset_id": dataset_id,
                    "config": {"configurable": {"dataset_id": dataset_id}},
                    "input": {"messages": [{"type": "human", "content": "分析一下"}]},
                },
            )
            responses[dataset_id] = response.text

    thread_1 = threading.Thread(target=_run, args=(dataset_id_1,))
    thread_2 = threading.Thread(target=_run, args=(dataset_id_2,))
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()

    assert f"dataset={dataset_id_1} current={dataset_id_1}" in responses[dataset_id_1]
    assert f"dataset={dataset_id_2} current={dataset_id_2}" in responses[dataset_id_2]


def test_ensure_preprocessed_only_runs_once_under_concurrency(client: TestClient, monkeypatch):
    dataset_id = _upload_dataset(client)
    dataset = data_manager.get_dataset(dataset_id)
    original_prepare = data_manager.prepare_analysis_dataframe
    call_count = 0
    call_count_lock = threading.Lock()
    release_event = threading.Event()
    first_call_started = threading.Event()

    def wrapped_prepare(raw_df, schema_profile):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
            current_count = call_count
        if current_count == 1:
            first_call_started.set()
            time.sleep(0.1)
            release_event.set()
        else:
            release_event.wait(timeout=1)
        return original_prepare(raw_df, schema_profile)

    monkeypatch.setattr(data_manager, "prepare_analysis_dataframe", wrapped_prepare)

    results: list[bool] = []

    def _ensure():
        ensured = data_manager.dataset_store.ensure_preprocessed(dataset_id)
        results.append(ensured.preprocessed)

    thread_1 = threading.Thread(target=_ensure)
    thread_2 = threading.Thread(target=_ensure)
    thread_1.start()
    first_call_started.wait(timeout=1)
    thread_2.start()
    thread_1.join()
    thread_2.join()

    assert dataset.dataset_id == dataset_id
    assert call_count == 1
    assert results == [True, True]


def test_agent_stream_wrapper_keeps_dataset_context_during_body_iteration():
    async def body_iterator():
        assert get_current_dataset_id() == "dataset-123"
        yield b"chunk-1"
        assert get_current_dataset_id() == "dataset-123"
        yield b"chunk-2"

    response = StreamingResponse(body_iterator(), media_type="text/plain")
    wrapped = server._bind_agent_response_iterator(response, "dataset-123")

    async def collect_body():
        chunks: list[bytes] = []
        async for chunk in wrapped.body_iterator:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect_body())
    assert chunks == [b"chunk-1", b"chunk-2"]
    assert get_current_dataset_id() is None
