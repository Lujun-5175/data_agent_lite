"""
Manual live integration test — uploads a CSV and asks various questions
to verify the backend works correctly.
"""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from typing import Any

import requests

BASE_URL = "http://127.0.0.1:8002"

# ── Telco-like dataset ──────────────────────────────────────
CSV_CONTENT = (
    "customerID,tenure,MonthlyCharges,TotalCharges,Churn,Contract,gender\n"
    "c1,1,29.85,29.85,No,Month-to-month,Female\n"
    "c2,34,56.95,1889.5,No,One year,Male\n"
    "c3,2,53.85,108.15,Yes,Month-to-month,Female\n"
    "c4,45,42.3,1840.75,No,Two year,Male\n"
    "c5,8,70.7,568.35,Yes,Month-to-month,Female\n"
    "c6,22,89.1,1949.4,No,One year,Male\n"
    "c7,10,29.75,301.9,Yes,Month-to-month,Female\n"
    "c8,60,109.9,6660.2,No,Two year,Male\n"
    "c9,5,79.35,401.45,Yes,Month-to-month,Female\n"
    "c10,72,99.65,7251.7,No,Two year,Male\n"
)


def upload_dataset() -> str:
    resp = requests.post(f"{BASE_URL}/upload", files={"file": ("telco.csv", CSV_CONTENT.encode("utf-8"), "text/csv")})
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    payload = resp.json()
    ds_id = payload["dataset_id"]
    print(f"[UPLOAD] dataset_id={ds_id}  rows={payload.get('row_count')}  cols={payload.get('column_count')}")
    return ds_id


def stream_chat(dataset_id: str, messages: list[dict[str, str]]) -> list[tuple[str, dict[str, Any]]]:
    body: dict[str, Any] = {
        "dataset_id": dataset_id,
        "config": {"configurable": {"dataset_id": dataset_id}},
        "input": {"messages": messages},
    }
    resp = requests.post(f"{BASE_URL}/chat/stream", json=body, stream=False)
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
        return [(f"http_error_{resp.status_code}", {"detail": resp.text[:200]})]

    events: list[tuple[str, dict[str, Any]]] = []
    for block in resp.text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        event_line = next((l for l in lines if l.startswith("event: ")), None)
        data_line = next((l for l in lines if l.startswith("data: ")), None)
        if event_line and data_line:
            event_type = event_line.replace("event: ", "", 1).strip()
            payload = json.loads(data_line.replace("data: ", "", 1))
            events.append((event_type, payload))
    return events


def get_text(events: list[tuple[str, dict[str, Any]]]) -> str:
    parts: list[str] = []
    for event_type, payload in events:
        if event_type == "message_chunk":
            content = payload.get("content", "")
            if isinstance(content, str):
                parts.append(content)
    return "".join(parts)


def get_errors(events: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [p for t, p in events if t == "error"]


def show_events_summary(events: list[tuple[str, dict[str, Any]]]) -> None:
    event_counts: dict[str, int] = {}
    for event_type, _ in events:
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    print(f"  Events: {event_counts}")
    errs = get_errors(events)
    if errs:
        for e in errs:
            print(f"  ERROR: {e.get('code', '?')}: {e.get('message', '?')[:120]}")
    text = get_text(events)
    if text:
        print(f"  Text: {text[:200]}{'...' if len(text) > 200 else ''}")


# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("MANUAL LIVE INTEGRATION TEST")
    print("=" * 60)

    # 1. Root health check
    print("\n--- 1. Root health check ---")
    r = requests.get(f"{BASE_URL}/")
    print(f"  {r.status_code}: {r.json()}")

    # 2. Upload dataset
    print("\n--- 2. Upload dataset ---")
    dataset_id = upload_dataset()

    # 3. Data preview
    print("\n--- 3. Data preview ---")
    r = requests.get(f"{BASE_URL}/data-preview", params={"dataset_id": dataset_id})
    if r.status_code == 200:
        data = r.json()
        print(f"  preview rows: {len(data.get('preview', []))}")
        print(f"  columns: {[c['name'] for c in data.get('columns', [])]}")
    else:
        print(f"  FAIL: {r.status_code} {r.text[:120]}")

    # 4. Stats query — describe numeric columns
    print("\n--- 4. Stats: describe numeric columns ---")
    events = stream_chat(dataset_id, [{"type": "human", "content": "请描述所有数值列的统计信息，包括均值、标准差、分位数等。"}])
    show_events_summary(events)

    # 5. Stats query — group summary
    print("\n--- 5. Stats: group summary (Churn rate by Contract) ---")
    events = stream_chat(dataset_id, [{"type": "human", "content": "按 Contract 分组计算 Churn 的占比（Yes rate），按占比降序排列。"}])
    show_events_summary(events)

    # 6. Python analysis — custom groupby
    print("\n--- 6. Python: total_amount (MonthlyCharges) by gender and Contract ---")
    events = stream_chat(dataset_id, [{"type": "human", "content": "用 python 计算不同 gender 和 Contract 组合的 MonthlyCharges 总和，按 gender 和 Contract 分组，打印结果。"}])
    show_events_summary(events)

    # 7. ML — logistic regression
    print("\n--- 7. ML: train logistic regression (Churn) ---")
    events = stream_chat(dataset_id, [{"type": "human", "content": "用逻辑回归预测 Churn，输出 accuracy 和特征重要性。"}])
    show_events_summary(events)

    # 8. Follow-up query
    print("\n--- 8. Follow-up: group summary ---")
    events = stream_chat(dataset_id, [{"type": "human", "content": "按 gender 分组计算 Churn 率。"}])
    show_events_summary(events)

    # 9. Error: no dataset (no dataset_id)
    print("\n--- 9. Error: request without dataset ---")
    resp = requests.post(
        f"{BASE_URL}/chat/stream",
        json={"input": {"messages": [{"type": "human", "content": "分析数据。"}]}},
    )
    print(f"  HTTP {resp.status_code}: {resp.json().get('error', {}).get('message', resp.text[:200])}")

    # 10. Delete dataset
    print("\n--- 10. Delete dataset ---")
    r = requests.delete(f"{BASE_URL}/datasets/{dataset_id}")
    print(f"  {r.status_code}: {r.json()}")

    print("\n" + "=" * 60)
    print("MANUAL TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
