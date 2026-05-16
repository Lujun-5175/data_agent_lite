from __future__ import annotations

import json

import pandas as pd

from src.ml_helpers import BaselineMLService
from src.result_normalizer import normalize_result_payload, summarize_tool_output
from src.result_types import artifact_registry, build_artifact
from src.tools import ReadOnlySeriesProxy


def test_normalize_pandas_series_json_safe():
    series = pd.Series([1.5, 2.75], name="value")

    normalized = normalize_result_payload(series)

    assert normalized["row_count"] == 2
    assert normalized["columns"] == ["index", "value"]
    assert normalized["rows"][0]["value"] == 1.5
    assert "ReadOnlySeriesProxy" not in json.dumps(normalized, ensure_ascii=False)


def test_normalize_dataframe_json_safe():
    frame = pd.DataFrame({"city": ["Shanghai", "Beijing"], "sales": [10, 20]})

    normalized = normalize_result_payload(frame)

    assert normalized["row_count"] == 2
    assert normalized["columns"] == ["city", "sales"]
    assert normalized["rows"][1]["sales"] == 20
    assert json.dumps(normalized, ensure_ascii=False)


def test_normalize_readonly_series_proxy_if_accessible():
    proxy = ReadOnlySeriesProxy(pd.Series([3, 4], name="score"))

    normalized = normalize_result_payload(proxy)

    assert normalized["row_count"] == 2
    assert normalized["rows"][0]["value"] == 3
    assert "ReadOnlySeriesProxy" not in json.dumps(normalized, ensure_ascii=False)


def test_row_count_does_not_count_dtype_line():
    text = "plan_type\nbasic       111\nstandard     90\npremium      39\ndtype: int64"

    payload = summarize_tool_output(text)

    assert payload is not None
    assert payload["row_count"] == 3
    assert len(payload["rows"]) == 3
    assert all("dtype" not in json.dumps(row, ensure_ascii=False).lower() for row in payload["rows"])


def test_live_prediction_result_does_not_include_proxy_repr():
    text = (
        "<src.tools.ReadOnlySeriesProxy object at 0x00000232735CCB40>\n"
        "---\n"
        "      segment  total  churned  churn_rate\n"
        "1  enterprise     54       18      0.3333\n"
        "2  mid-market     33       10      0.3030\n"
        "3         smb     55       14      0.2545\n"
        "0    consumer     98       24      0.2449\n"
    )

    payload = summarize_tool_output(text)

    assert payload is not None
    assert payload["row_count"] == 4
    assert "ReadOnlySeriesProxy" not in json.dumps(payload, ensure_ascii=False)
    assert "dtype" not in json.dumps(payload, ensure_ascii=False).lower()


def test_ml_artifact_reuse_latest_accepts_model_alias():
    dataset_id = "reuse-state-test"
    try:
        artifact_registry.clear_dataset(dataset_id)
        artifact_registry.register(
            dataset_id,
            build_artifact(
                artifact_type="model_result",
                dataset_id=dataset_id,
                payload={
                    "model_type": "logistic_regression",
                    "metrics": {"accuracy": 0.9},
                },
            ),
        )

        service = BaselineMLService(dataset_id=dataset_id)
        latest = service.latest(artifact_type="model")

        assert latest["artifact_type"] == "model_result"
        assert latest["model_type"] == "logistic_regression"
        assert latest["metrics"]["accuracy"] == 0.9
    finally:
        artifact_registry.clear_dataset(dataset_id)
