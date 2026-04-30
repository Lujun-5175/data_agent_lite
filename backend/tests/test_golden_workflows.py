from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import server
from src.data_manager import get_dataframe
from src.errors import AppError
from src.tools import MLHelperAPI, ProfileHelperAPI, StatsHelperAPI, set_current_image_event


def _extract_messages(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    messages = inputs.get("messages", [])
    return [item for item in messages if isinstance(item, dict)]


def _latest_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("type") in {"human", "user"}:
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _text_event(content: str) -> dict[str, Any]:
    return {"event": "on_chat_model_stream", "data": {"chunk": content}, "name": "golden-model"}


class GoldenWorkflowGraph:
    async def astream_events(self, inputs: dict[str, Any], config: dict[str, Any], context: Any, version: str):
        dataset_id = config.get("configurable", {}).get("dataset_id")
        if not dataset_id:
            raise AppError("dataset_required", "当前未选择数据集，请先上传 CSV 文件后再进行数据分析。", 400)

        df = get_dataframe(str(dataset_id))
        stats = StatsHelperAPI(df, dataset_id=str(dataset_id))
        profile = ProfileHelperAPI(dataset_id=str(dataset_id))
        ml = MLHelperAPI(df, dataset_id=str(dataset_id))
        messages = _extract_messages(inputs)
        latest = _latest_user_message(messages).strip().lower()
        previous_human = " ".join(
            str(message.get("content", "")).lower()
            for message in messages[:-1]
            if message.get("type") in {"human", "user"}
        )

        if "profit" in latest or "nonexistent" in latest:
            raise AppError("dataset_load_error", "指定的列不存在。", 400)
        if "customerriskscore" in latest:
            raise AppError("dataset_load_error", "指定的列不存在。", 400)

        yield {"event": "on_tool_start", "name": "python_inter", "data": {}}

        if "total" in latest and "count" in latest and "average" in latest:
            payload = {
                "total_sales": float(df["sales"].sum()),
                "row_count": int(len(df.index)),
                "avg_sales": round(float(df["sales"].mean()), 4),
            }
            yield _text_event(json.dumps(payload, ensure_ascii=False))
        elif "average sales for us users grouped by state" in latest:
            grouped = (
                df[df["country"] == "US"]
                .groupby("state", dropna=False)["sales"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index(name="avg_sales")
            )
            payload = {
                "grouped": grouped.to_dict(orient="records"),
                "row_count": int(len(grouped.index)),
            }
            yield _text_event(json.dumps(payload, ensure_ascii=False))
        elif "top 3 users by sales" in latest:
            rows = (
                df.sort_values("sales", ascending=False)
                .head(3)[["user_id", "sales", "state"]]
                .to_dict(orient="records")
            )
            payload = {"top_n": 3, "rows": rows}
            yield _text_event(json.dumps(payload, ensure_ascii=False))
        elif "monthly sales aggregation" in latest:
            target = df.copy(deep=True)
            target["month"] = pd.to_datetime(target["date"]).dt.to_period("M").astype(str)
            grouped = (
                target.groupby("month", dropna=False)["sales"]
                .sum()
                .reset_index(name="sales_sum")
                .sort_values("month")
            )
            payload = {"granularity": "month", "rows": grouped.to_dict(orient="records")}
            yield _text_event(json.dumps(payload, ensure_ascii=False))
        elif "bar chart of sales by state for us" in latest:
            filename = f"{uuid4().hex}.png"
            image_path = (server.IMAGES_DIR / filename).resolve()
            image_path.write_bytes(b"PNG")
            set_current_image_event(
                {
                    "type": "image_generated",
                    "filename": filename,
                    "tool_name": "fig_inter",
                }
            )
            yield _text_event("已根据 US 用户按州生成柱状图。")
            yield {"event": "on_tool_end", "name": "fig_inter", "data": {}}
            return
        elif "now only look at california" in latest:
            metric = "total_sales"
            if "total sales" not in latest and "total sales" in previous_human:
                metric = "total_sales"
            scope_df = df[(df["country"] == "US") & (df["state"] == "California")]
            payload = {
                "scope": "California",
                "inherited_metric": metric,
                "total_sales": float(scope_df["sales"].sum()),
                "row_count": int(len(scope_df.index)),
            }
            yield _text_event(json.dumps(payload, ensure_ascii=False))
        elif "describe numeric columns" in latest:
            artifact = stats.describe_numeric(["sales", "orders"])
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
        elif "churn rate by state" in latest:
            artifact = stats.group_summary(
                group_by="state",
                metrics=[{"op": "rate", "column": "churn", "as": "churn_rate"}, {"op": "count", "as": "row_count"}],
                sort_by="churn_rate",
                ascending=False,
                top_n=10,
            )
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
        elif "chi-square between contract and churn" in latest:
            artifact = stats.chi_square("contract", "churn")
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
        elif "show latest stats artifact" in latest:
            artifact = stats.latest()
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
        elif "inspect dataset schema" in latest or "数据结构" in latest or "schema profile" in latest:
            artifact = profile.schema()
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
        elif "likely targets" in latest or "目标候选" in latest:
            artifact = profile.model_prep_plan()
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
        elif "preprocessing" in latest or "预处理" in latest:
            artifact = profile.analysis_preprocess()
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
        elif "unsuitable for modeling" in latest or "不适合建模" in latest:
            artifact = profile.model_prep_plan()
            payload = {
                "artifact_type": "model_prep_plan",
                "excluded_columns": artifact.get("excluded_columns", []),
                "warnings": artifact.get("warnings", []),
            }
            yield _text_event(json.dumps(payload, ensure_ascii=False))
        elif "train baseline logistic model" in latest or "训练 baseline 逻辑回归" in latest:
            yield {"event": "on_tool_start", "name": "ml_execute", "data": {"action": "train"}}
            artifact = ml.logistic_fit(target="Churn", positive_label="Yes")
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
            yield {"event": "on_tool_end", "name": "ml_execute", "data": {"action": "train"}}
        elif "show model metrics" in latest or "查看模型指标" in latest:
            yield {"event": "on_tool_start", "name": "ml_execute", "data": {"action": "metrics"}}
            artifact = ml.metrics()
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
            yield {"event": "on_tool_end", "name": "ml_execute", "data": {"action": "metrics"}}
        elif "show feature importance" in latest or "特征重要性" in latest:
            yield {"event": "on_tool_start", "name": "ml_execute", "data": {"action": "feature_importance"}}
            artifact = ml.feature_importance(top_k=5)
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
            yield {"event": "on_tool_end", "name": "ml_execute", "data": {"action": "feature_importance"}}
        elif "train baseline linear model" in latest or "训练 baseline 线性回归" in latest:
            yield {"event": "on_tool_start", "name": "ml_execute", "data": {"action": "train"}}
            artifact = ml.linear_regression_fit(target="TotalCharges")
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
            yield {"event": "on_tool_end", "name": "ml_execute", "data": {"action": "train"}}
        elif "train model with missing target" in latest or "用不存在目标列训练模型" in latest:
            raise AppError("invalid_python_code", "目标列不存在: MissingTarget", 400)
        elif ("tenure" in latest and "monthlycharges" in latest and "totalcharges" in latest) and (
            "数值摘要统计" in latest or "numeric summary" in latest or "描述" in latest
        ):
            numeric_artifact = stats.describe_numeric(["tenure", "MonthlyCharges", "TotalCharges"])
            grouped_artifact = stats.group_summary(
                group_by="Churn",
                metrics=[
                    {"op": "mean", "column": "tenure", "as": "tenure_mean"},
                    {"op": "median", "column": "tenure", "as": "tenure_median"},
                    {"op": "mean", "column": "MonthlyCharges", "as": "monthlycharges_mean"},
                    {"op": "median", "column": "MonthlyCharges", "as": "monthlycharges_median"},
                    {"op": "count", "as": "sample_size"},
                ],
                sort_by="group",
                ascending=True,
                top_n=10,
            )
            combined = {
                "artifact_type": "stats_result",
                "stats_type": "combined_summary",
                "numeric_artifact": numeric_artifact,
                "grouped_artifact": grouped_artifact,
            }
            yield _text_event(json.dumps(combined, ensure_ascii=False))
        elif "monthlycharges" in latest and ("t-test" in latest or "t test" in latest or "t检验" in latest):
            artifact = stats.t_test("MonthlyCharges", "Churn", "Yes", "No")
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
        elif "contract" in latest and ("chi-square" in latest or "chi square" in latest or "卡方" in latest):
            artifact = stats.chi_square("Contract", "Churn")
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
        elif "tenure" in latest and "monthlycharges" in latest and "totalcharges" in latest and ("相关性" in latest or "correlation" in latest):
            artifact = stats.correlation(["tenure", "MonthlyCharges", "TotalCharges"], top_k=3)
            yield _text_event(json.dumps(artifact, ensure_ascii=False))
        elif "最近一次统计结果" in latest or "latest artifact" in latest:
            latest_artifact = stats.latest()
            summary = {
                "artifact_type": "stats_result",
                "stats_type": "latest_summary",
                "based_on_artifact_id": latest_artifact["artifact_id"],
                "based_on_stats_type": latest_artifact.get("stats_type", latest_artifact.get("test_type")),
                "description_vs_tested": {
                    "descriptive": "已包含分组描述/相关性等描述性结果。",
                    "tested": "若最近结果为 test_result，则包含显著性检验结论。",
                    "causal_note": "相关或显著不代表因果。"
                },
            }
            yield _text_event(json.dumps(summary, ensure_ascii=False))
        elif "state = florida" in latest or "florida" in latest:
            payload = {"row_count": 0, "message": "未找到匹配记录"}
            yield _text_event(json.dumps(payload, ensure_ascii=False))
        else:
            payload = {"message": "golden test fallback"}
            yield _text_event(json.dumps(payload, ensure_ascii=False))

        yield {"event": "on_tool_end", "name": "python_inter", "data": {}}


@pytest.fixture(autouse=True)
def golden_graph(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "graph", GoldenWorkflowGraph())


def _upload_fixture_dataset(client: TestClient) -> str:
    csv_content = (
        "date,country,state,user_id,sales,orders,churn,contract\n"
        "2024-01-01,US,California,u1,100,1,yes,monthly\n"
        "2024-01-01,US,New York,u2,200,2,no,yearly\n"
        "2024-01-02,US,California,u3,300,3,yes,monthly\n"
        "2024-01-02,US,Texas,u4,50,1,no,monthly\n"
        "2024-01-08,US,California,u5,120,1,no,yearly\n"
        "2024-02-01,US,New York,u6,80,1,no,yearly\n"
        "2024-02-02,CA,Ontario,u7,500,5,no,yearly\n"
        "2024-02-03,US,California,u8,400,4,yes,monthly\n"
    ).encode("utf-8")

    response = client.post("/upload", files={"file": ("golden.csv", csv_content, "text/csv")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"]
    return str(payload["dataset_id"])


def _upload_telco_fixture_dataset(client: TestClient) -> str:
    csv_content = (
        "customerID,tenure,MonthlyCharges,TotalCharges,Churn,Contract\n"
        "c1,1,29.85,29.85,No,Month-to-month\n"
        "c2,34,56.95,1889.5,No,One year\n"
        "c3,2,53.85,108.15,Yes,Month-to-month\n"
        "c4,45,42.3,1840.75,No,Two year\n"
        "c5,8,70.7,568.35,Yes,Month-to-month\n"
        "c6,22,89.1,1949.4,No,One year\n"
        "c7,10,29.75,301.9,Yes,Month-to-month\n"
        "c8,60,109.9,6660.2,No,Two year\n"
        "c9,5,79.35,401.45,Yes,Month-to-month\n"
        "c10,72,99.65,7251.7,No,Two year\n"
    ).encode("utf-8")
    response = client.post("/upload", files={"file": ("telco.csv", csv_content, "text/csv")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"]
    return str(payload["dataset_id"])


def _upload_regression_fixture_dataset(client: TestClient) -> str:
    rows = ["customerID,tenure,MonthlyCharges,TotalCharges,Contract"]
    for idx in range(1, 121):
        tenure = 1 + (idx % 24)
        monthly = 20 + (idx % 15) * 4
        contract = "monthly" if idx % 2 == 0 else "yearly"
        total = monthly * tenure * 0.9 + (6 if contract == "yearly" else 0)
        rows.append(f"u{idx},{tenure},{monthly},{total},{contract}")
    csv_content = ("\n".join(rows) + "\n").encode("utf-8")
    response = client.post("/upload", files={"file": ("regression.csv", csv_content, "text/csv")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"]
    return str(payload["dataset_id"])


def _chat_stream_events(client: TestClient, dataset_id: str, messages: list[dict[str, str]]) -> list[tuple[str, dict[str, Any]]]:
    response = client.post(
        "/chat/stream",
        json={
            "dataset_id": dataset_id,
            "config": {"configurable": {"dataset_id": dataset_id}},
            "input": {"messages": messages},
        },
    )
    assert response.status_code == 200
    blocks = [block for block in response.text.split("\n\n") if block.strip()]
    events: list[tuple[str, dict[str, Any]]] = []
    for block in blocks:
        lines = block.splitlines()
        event_line = next((line for line in lines if line.startswith("event: ")), None)
        data_line = next((line for line in lines if line.startswith("data: ")), None)
        assert event_line is not None
        assert data_line is not None
        event_type = event_line.replace("event: ", "", 1).strip()
        payload = json.loads(data_line.replace("data: ", "", 1))
        events.append((event_type, payload))
    return events


def _latest_chunk_json(events: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    chunks = [payload for event_type, payload in events if event_type == "message_chunk"]
    assert chunks
    content = chunks[-1]["content"]
    assert isinstance(content, str)
    return json.loads(content)


def test_golden_basic_metric_query(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Please return total sales, count and average sales."}],
    )
    result = _latest_chunk_json(events)
    assert result["total_sales"] == 1750.0
    assert result["row_count"] == 8
    assert result["avg_sales"] == 218.75


def test_golden_filter_and_aggregation(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Show average sales for US users grouped by state, top 10"}],
    )
    result = _latest_chunk_json(events)
    rows = result["grouped"]
    assert result["row_count"] == 3
    assert [row["state"] for row in rows] == ["California", "New York", "Texas"]
    assert [row["avg_sales"] for row in rows] == [230.0, 140.0, 50.0]


def test_golden_sort_top_n(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Show top 3 users by sales"}],
    )
    result = _latest_chunk_json(events)
    assert result["top_n"] == 3
    sales = [row["sales"] for row in result["rows"]]
    assert sales == sorted(sales, reverse=True)
    assert len(result["rows"]) == 3


def test_golden_time_grouping_monthly(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Give me monthly sales aggregation"}],
    )
    result = _latest_chunk_json(events)
    assert result["granularity"] == "month"
    assert result["rows"] == [
        {"month": "2024-01", "sales_sum": 770},
        {"month": "2024-02", "sales_sum": 980},
    ]


def test_golden_chart_generation(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Please create a bar chart of sales by state for US"}],
    )
    image_events = [payload for event_type, payload in events if event_type == "image_generated"]
    assert image_events
    image = image_events[-1]
    filename = image["filename"]
    assert isinstance(filename, str) and filename.endswith(".png")
    assert "/" not in filename and "\\" not in filename
    image_path = (server.IMAGES_DIR / filename).resolve()
    assert image_path.exists()
    assert image_path.parent == server.IMAGES_DIR.resolve()
    assert image["image_url"].endswith(f"/static/images/{filename}")


def test_golden_follow_up_query_context(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [
            {"type": "human", "content": "Show total sales for US users"},
            {"type": "assistant", "content": "US total sales computed."},
            {"type": "human", "content": "Now only look at California"},
        ],
    )
    result = _latest_chunk_json(events)
    assert result["scope"] == "California"
    assert result["inherited_metric"] == "total_sales"
    assert result["row_count"] == 4
    assert result["total_sales"] == 920.0


def test_golden_error_path_invalid_column(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Please analyze profit column and correlation"}],
    )
    errors = [payload for event_type, payload in events if event_type == "error"]
    assert errors
    error = errors[-1]
    assert error["code"] == "dataset_load_error"
    assert "列不存在" in error["message"]


def test_golden_empty_result_behavior(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Filter state = Florida and summarize"}],
    )
    result = _latest_chunk_json(events)
    assert result["row_count"] == 0
    assert "未找到" in result["message"]


def test_golden_stats_query_returns_structured_artifact(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Please describe numeric columns"}],
    )
    result = _latest_chunk_json(events)
    assert result["artifact_type"] == "stats_result"
    assert result["stats_type"] == "describe_numeric"
    assert "rows" in result and len(result["rows"]) == 2


def test_golden_stats_test_query_returns_structured_artifact(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Run chi-square between contract and churn"}],
    )
    result = _latest_chunk_json(events)
    assert result["artifact_type"] == "test_result"
    assert result["test_type"] == "chi_square"
    assert "p_value" in result


def test_golden_stats_follow_up_reuses_latest_artifact(client: TestClient):
    dataset_id = _upload_fixture_dataset(client)
    first_events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Show churn rate by state"}],
    )
    first_result = _latest_chunk_json(first_events)
    assert first_result["artifact_type"] == "stats_result"
    assert first_result["stats_type"] == "group_summary"

    follow_up_events = _chat_stream_events(
        client,
        dataset_id,
        [
            {"type": "human", "content": "Show churn rate by state"},
            {"type": "assistant", "content": "已经完成分组统计。"},
            {"type": "human", "content": "Show latest stats artifact"},
        ],
    )
    result = _latest_chunk_json(follow_up_events)
    assert result["artifact_type"] == "stats_result"
    assert result["stats_type"] == "group_summary"
    assert result["row_count"] > 0


def test_golden_telco_stats_first_turn_summary(client: TestClient):
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [
            {
                "type": "human",
                "content": "请先做 tenure、MonthlyCharges、TotalCharges 的数值摘要统计，并按 Churn 分组给均值/中位数/样本量",
            }
        ],
    )
    result = _latest_chunk_json(events)
    assert result["artifact_type"] == "stats_result"
    assert result["stats_type"] == "combined_summary"
    assert result["numeric_artifact"]["stats_type"] == "describe_numeric"
    assert result["grouped_artifact"]["stats_type"] == "group_summary"


def test_golden_telco_follow_up_t_test(client: TestClient):
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "对 MonthlyCharges 在 Churn=Yes 和 Churn=No 两组做 t-test"}],
    )
    result = _latest_chunk_json(events)
    assert result["artifact_type"] == "test_result"
    assert result["test_type"] == "t_test"
    assert "p_value" in result
    assert result["group_a_size"] > 0 and result["group_b_size"] > 0


def test_golden_telco_follow_up_chi_square(client: TestClient):
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "检验 Contract 和 Churn 的卡方关联（chi-square）"}],
    )
    result = _latest_chunk_json(events)
    assert result["artifact_type"] == "test_result"
    assert result["test_type"] == "chi_square"
    assert set(["statistic", "p_value", "dof"]).issubset(result.keys())


def test_golden_telco_correlation_and_latest_artifact(client: TestClient):
    dataset_id = _upload_telco_fixture_dataset(client)
    corr_events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "看一下 tenure、MonthlyCharges、TotalCharges 的相关性"}],
    )
    corr_result = _latest_chunk_json(corr_events)
    assert corr_result["artifact_type"] == "stats_result"
    assert corr_result["stats_type"] == "correlation"
    assert len(corr_result["matrix"]) == 3
    assert corr_result["top_pairs"]
    latest_artifact_id = corr_result["artifact_id"]

    follow_events = _chat_stream_events(
        client,
        dataset_id,
        [
            {"type": "human", "content": "看一下 tenure、MonthlyCharges、TotalCharges 的相关性"},
            {"type": "assistant", "content": "相关性已完成。"},
            {"type": "human", "content": "请基于最近一次统计结果做总结，不要重新跑无关分析"},
        ],
    )
    follow_result = _latest_chunk_json(follow_events)
    assert follow_result["artifact_type"] == "stats_result"
    assert follow_result["stats_type"] == "latest_summary"
    assert follow_result["based_on_artifact_id"] == latest_artifact_id


def test_golden_telco_error_nonexistent_column(client: TestClient):
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "请对不存在字段 CustomerRiskScore 按 Churn 分组并检验显著性"}],
    )
    errors = [payload for event_type, payload in events if event_type == "error"]
    assert errors
    error = errors[-1]
    assert error["code"] == "dataset_load_error"
    assert "列不存在" in error["message"]


def test_golden_schema_inspection_query(client: TestClient):
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Please inspect dataset schema and show schema profile"}],
    )
    result = _latest_chunk_json(events)
    assert result["artifact_type"] == "schema_profile"
    assert "columns" in result and result["columns"]


def test_golden_target_candidate_query(client: TestClient):
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Tell me likely targets for this dataset"}],
    )
    result = _latest_chunk_json(events)
    assert result["artifact_type"] == "model_prep_plan"
    assert "target_status" in result
    assert "candidate_features" in result


def test_golden_preprocessing_query(client: TestClient):
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "What preprocessing did you apply for analysis?"}],
    )
    result = _latest_chunk_json(events)
    assert result["artifact_type"] == "preprocess_result"
    assert result["stage"] == "analysis"
    assert "steps" in result


def test_golden_ml_logistic_and_follow_up_metrics_and_importance(client: TestClient):
    dataset_id = _upload_telco_fixture_dataset(client)
    train_events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Please train baseline logistic model for churn"}],
    )
    train_result = _latest_chunk_json(train_events)
    assert train_result["artifact_type"] == "model_result"
    assert train_result["model_type"] == "logistic_regression"

    metrics_events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Please show model metrics"}],
    )
    metrics_result = _latest_chunk_json(metrics_events)
    assert metrics_result["artifact_type"] == "metrics_result"
    assert metrics_result["source_model_artifact_id"] == train_result["artifact_id"]

    fi_events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Please show feature importance"}],
    )
    fi_result = _latest_chunk_json(fi_events)
    assert fi_result["artifact_type"] == "feature_importance_result"
    assert fi_result["source_model_artifact_id"] == train_result["artifact_id"]


def test_golden_ml_linear_regression_flow(client: TestClient):
    dataset_id = _upload_regression_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "Please train baseline linear model on TotalCharges"}],
    )
    result = _latest_chunk_json(events)
    assert result["artifact_type"] == "model_result"
    assert result["model_type"] == "linear_regression"
    assert set(result["metrics"].keys()) == {"rmse", "mae", "r2"}


def test_golden_ml_invalid_target_error(client: TestClient):
    dataset_id = _upload_telco_fixture_dataset(client)
    events = _chat_stream_events(
        client,
        dataset_id,
        [{"type": "human", "content": "train model with missing target"}],
    )
    errors = [payload for event_type, payload in events if event_type == "error"]
    assert errors
    assert errors[-1]["code"] in {"invalid_python_code", "dataset_load_error"}
