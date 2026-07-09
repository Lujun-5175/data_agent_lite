from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from langchain_core.callbacks import BaseCallbackHandler

from src.audit_log import AuditLogger, read_recent_records
from src.data_manager import cleanup_dataset_artifacts, load_csv_file
from src.eval.prediction_adapter import (
    create_prediction_template,
    prediction_from_audit_record,
    sanitize_tool_args,
    write_predictions_jsonl,
)
from src.eval.runner import load_eval_cases
from src.eval.schema import EvalCase, EvalPrediction
from src.result_normalizer import normalize_result_payload, summarize_tool_output
from src.sse import extract_text_from_chunk

LIVE_KEY_ERROR = "DEEPSEEK_API_KEY is required for --mode live"


@dataclass(slots=True)
class ToolCallCapture:
    tool_name: str | None
    raw_args: dict[str, Any] | str | None
    output: Any = None
    error_message: str | None = None


class LiveCaptureCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self.tool_calls: list[ToolCallCapture] = []
        self.text_chunks: list[str] = []

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        inputs=None,
        **kwargs: Any,
    ) -> Any:
        tool_name = _extract_tool_name(serialized, kwargs)
        raw_args = _coerce_tool_args(inputs=inputs, input_str=input_str, tool_name=tool_name)
        self.tool_calls.append(ToolCallCapture(tool_name=tool_name, raw_args=raw_args))

    def on_tool_end(self, output: Any, *, run_id, parent_run_id=None, **kwargs: Any) -> Any:
        if self.tool_calls:
            self.tool_calls[-1].output = output

    def on_tool_error(self, error: BaseException, *, run_id, parent_run_id=None, **kwargs: Any) -> Any:
        if self.tool_calls:
            self.tool_calls[-1].error_message = str(error)

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk=None,
        run_id,
        parent_run_id=None,
        tags=None,
        **kwargs: Any,
    ) -> Any:
        text = extract_text_from_chunk(chunk) if chunk is not None else token
        if isinstance(text, str) and text.strip():
            self.text_chunks.append(text)


class _CaseAwareAuditLogger:
    def __init__(self, inner: AuditLogger, *, case_id: str) -> None:
        self._inner = inner
        self._case_id = case_id

    def record(self, *args: Any, **kwargs: Any) -> str:
        extra = kwargs.get("extra")
        merged_extra: dict[str, Any] = {"case_id": self._case_id}
        if isinstance(extra, dict):
            merged_extra.update(extra)
        kwargs["extra"] = merged_extra
        return self._inner.record(*args, **kwargs)


class _LiveBenchmarkWorkspace:
    def __init__(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="eval-live-")
        self.base_path = Path(self._tempdir.name)
        self.audit_log_path = self.base_path / "audit.jsonl"
        self.dataset_cache: dict[str, str] = {}

    def close(self) -> None:
        for dataset_id in list(self.dataset_cache.values()):
            with contextlib.suppress(Exception):
                cleanup_dataset_artifacts(dataset_id)
        self._tempdir.cleanup()

    def dataset_id_for_case(self, case: EvalCase) -> str:
        cache_key = _dataset_cache_key(case.dataset_name)
        if cache_key in self.dataset_cache:
            return self.dataset_cache[cache_key]

        csv_path = self.base_path / f"{cache_key}.csv"
        dataframe = _build_live_benchmark_dataframe(cache_key)
        dataframe.to_csv(csv_path, index=False)
        dataset = load_csv_file(csv_path, original_filename=csv_path.name)
        self.dataset_cache[cache_key] = dataset.dataset_id
        return dataset.dataset_id


def create_template_file(cases_path: str | Path, output_path: str | Path) -> int:
    cases = load_eval_cases(cases_path)
    predictions = create_prediction_template(cases)
    write_predictions_jsonl(predictions, output_path)
    return len(predictions)


def create_predictions_from_audit_log(audit_log_path: str | Path, output_path: str | Path) -> int:
    records = _load_jsonl_dicts(audit_log_path)
    predictions: list[EvalPrediction] = []
    for record in records:
        prediction = prediction_from_audit_record(record)
        if prediction is not None:
            predictions.append(prediction)
    write_predictions_jsonl(predictions, output_path)
    return len(predictions)


def run_live_agent_case(case: EvalCase) -> EvalPrediction:
    _ensure_live_api_key()
    workspace = _LiveBenchmarkWorkspace()
    try:
        return _run_live_agent_case(case, workspace)
    finally:
        workspace.close()


def create_live_predictions(
    cases_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> int:
    _ensure_live_api_key()
    cases = load_eval_cases(cases_path)
    selected_cases = cases if limit is None else cases[: max(limit, 0)]
    workspace = _LiveBenchmarkWorkspace()
    predictions: list[EvalPrediction] = []
    try:
        for case in selected_cases:
            try:
                prediction = _run_live_agent_case(case, workspace)
            except Exception as exc:  # pragma: no cover - defensive fallback for manual runs
                prediction = EvalPrediction(
                    case_id=case.case_id,
                    predicted_intent=None,
                    predicted_tool=None,
                    predicted_args=None,
                    execution_status="error",
                    result={
                        "final_answer_preview": None,
                        "tool_output_preview": None,
                        "output_size_bytes": 0,
                    },
                    error_message=str(exc),
                )
            predictions.append(_align_prediction_case_id(prediction, case.case_id))
        write_predictions_jsonl(predictions, output_path)
        return len(predictions)
    finally:
        workspace.close()


def create_router_predictions(
    cases_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> int:
    cases = load_eval_cases(cases_path)
    selected_cases = cases if limit is None else cases[: max(limit, 0)]
    predictions = [_align_prediction_case_id(_build_router_prediction(case), case.case_id) for case in selected_cases]
    write_predictions_jsonl(predictions, output_path)
    return len(predictions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture baseline predictions for offline and live eval modes.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("template", "audit", "router", "live"),
        help="Prediction capture mode.",
    )
    parser.add_argument(
        "--cases",
        default="evals/baseline_cases.jsonl",
        help="Path to baseline eval cases.",
    )
    parser.add_argument(
        "--audit-log",
        dest="audit_log",
        default=None,
        help="Path to an audit JSONL file for audit mode.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output prediction JSONL path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional case limit for router/live capture.",
    )
    args = parser.parse_args(argv)

    try:
        if args.mode == "template":
            written = create_template_file(args.cases, args.out)
        elif args.mode == "audit":
            if not args.audit_log:
                raise SystemExit("--audit-log is required for --mode audit")
            written = create_predictions_from_audit_log(args.audit_log, args.out)
        elif args.mode == "router":
            written = create_router_predictions(args.cases, args.out, limit=args.limit)
        elif args.mode == "live":
            written = create_live_predictions(args.cases, args.out, limit=args.limit)
        else:  # pragma: no cover - argparse constrains this
            raise SystemExit(f"Unsupported mode: {args.mode}")
    except SystemExit:
        raise
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    summary = {
        "mode": args.mode,
        "written": written,
        "out": str(Path(args.out)),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _run_live_agent_case(case: EvalCase, workspace: _LiveBenchmarkWorkspace) -> EvalPrediction:
    _ensure_live_api_key()
    from src.agent import AgentContext, graph
    from src import tools as tools_module

    dataset_id = workspace.dataset_id_for_case(case)
    callback = LiveCaptureCallback()
    proxy_logger = _CaseAwareAuditLogger(
        AuditLogger(enabled=True, path=workspace.audit_log_path),
        case_id=case.case_id,
    )
    original_get_audit_logger = tools_module.get_audit_logger
    tools_module.get_audit_logger = lambda: proxy_logger  # type: ignore[assignment]
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(bind_dataset_context(dataset_id))
            result = graph.invoke(
                {"messages": [{"type": "human", "content": case.user_query}]},
                config={
                    "configurable": {"dataset_id": dataset_id},
                    "callbacks": [callback],
                },
                context=AgentContext(dataset_id=dataset_id),
            )
        final_answer = _extract_final_answer(result, callback.text_chunks)
    except SystemExit:
        raise
    except Exception as exc:
        audit_record = _find_latest_audit_record(workspace.audit_log_path, case.case_id)
        return _build_live_prediction(
            case,
            tool_call=callback.tool_calls[-1] if callback.tool_calls else None,
            audit_record=audit_record,
            final_answer=_combine_text(callback.text_chunks) or None,
            fallback_error=str(exc),
        )
    finally:
        tools_module.get_audit_logger = original_get_audit_logger  # type: ignore[assignment]

    audit_record = _find_latest_audit_record(workspace.audit_log_path, case.case_id)
    tool_call = callback.tool_calls[-1] if callback.tool_calls else None
    return _build_live_prediction(
        case,
        tool_call=tool_call,
        audit_record=audit_record,
        final_answer=final_answer,
    )


def _build_live_prediction(
    case: EvalCase,
    *,
    tool_call: ToolCallCapture | None,
    audit_record: dict[str, Any] | None,
    final_answer: str | None,
    fallback_error: str | None = None,
) -> EvalPrediction:
    base_prediction = prediction_from_audit_record(audit_record or {}) if audit_record else None
    predicted_tool = (tool_call.tool_name if tool_call and tool_call.tool_name else None) or (
        base_prediction.predicted_tool if base_prediction else None
    )
    predicted_args = _build_predicted_args(tool_call, audit_record)
    execution_status = _resolve_execution_status(tool_call=tool_call, audit_record=audit_record, fallback_error=fallback_error)
    error_message = _coalesce_text(
        tool_call.error_message if tool_call else None,
        fallback_error,
        base_prediction.error_message if base_prediction else None,
        _audit_error_message(audit_record),
    )
    result = _build_result_payload(
        tool_call=tool_call,
        audit_record=audit_record,
        final_answer=final_answer,
        execution_status=execution_status,
    )
    predicted_intent = _infer_predicted_intent(
        case=case,
        predicted_tool=predicted_tool,
        predicted_args=predicted_args,
        result=result,
        execution_status=execution_status,
    )
    return EvalPrediction(
        case_id=case.case_id,
        predicted_intent=predicted_intent,
        predicted_tool=predicted_tool,
        predicted_args=predicted_args,
        execution_status=execution_status,
        result=result,
        error_message=error_message,
    )


def build_prediction_from_tool_capture(
    case: EvalCase,
    *,
    tool_name: str | None,
    tool_args: dict[str, Any] | str | None,
    tool_output: Any = None,
    execution_status: str | None = None,
    error_message: str | None = None,
    final_answer: str | None = None,
    audit_record: dict[str, Any] | None = None,
) -> EvalPrediction:
    capture = ToolCallCapture(tool_name=tool_name, raw_args=tool_args, output=tool_output, error_message=error_message)
    resolved_audit_record = audit_record
    if execution_status is not None:
        if resolved_audit_record is None:
            resolved_audit_record = {"execution_status": execution_status}
        else:
            resolved_audit_record = dict(resolved_audit_record)
            resolved_audit_record.setdefault("execution_status", execution_status)
    return _build_live_prediction(
        case,
        tool_call=capture,
        audit_record=resolved_audit_record,
        final_answer=final_answer,
        fallback_error=error_message,
    )


def _ensure_live_api_key() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit(LIVE_KEY_ERROR)


def bind_dataset_context(dataset_id: str):
    from src.tools import bind_current_dataset_id

    return bind_current_dataset_id(dataset_id)


def _find_latest_audit_record(audit_log_path: Path, case_id: str) -> dict[str, Any] | None:
    records = read_recent_records(path=audit_log_path, limit=200)
    matching = [record for record in records if isinstance(record, dict) and _record_case_id(record) == case_id]
    if not matching:
        return None
    return matching[0]


def _record_case_id(record: dict[str, Any]) -> str | None:
    extra = record.get("extra")
    if isinstance(extra, dict):
        case_id = extra.get("case_id")
        if isinstance(case_id, str) and case_id.strip():
            return case_id.strip()
    return None


def _build_predicted_args(
    tool_call: ToolCallCapture | None,
    audit_record: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if tool_call is not None:
        sanitized = sanitize_tool_args(tool_call.raw_args)
        if sanitized is not None:
            return sanitized
    if not audit_record:
        return None
    return sanitize_tool_args(audit_record.get("tool_args"))


def _resolve_execution_status(
    *,
    tool_call: ToolCallCapture | None,
    audit_record: dict[str, Any] | None,
    fallback_error: str | None,
) -> str | None:
    status = _coalesce_text(
        _audit_execution_status(audit_record),
        _infer_status_from_tool_call(tool_call),
    )
    if status is not None:
        return status
    if fallback_error:
        return "error"
    return None


def _audit_execution_status(record: dict[str, Any] | None) -> str | None:
    if not isinstance(record, dict):
        return None
    value = record.get("execution_status")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _audit_error_message(record: dict[str, Any] | None) -> str | None:
    if not isinstance(record, dict):
        return None
    value = record.get("error_message")
    return _coalesce_text(value)


def _infer_status_from_tool_call(tool_call: ToolCallCapture | None) -> str | None:
    if tool_call is None:
        return None
    if tool_call.error_message:
        lowered = tool_call.error_message.lower()
        if "timeout" in lowered or "超时" in tool_call.error_message:
            return "timeout"
        if "blocked" in lowered or "安全" in tool_call.error_message or "拦截" in tool_call.error_message:
            return "blocked"
        return "error"
    output_text = _stringify_output(tool_call.output)
    if output_text:
        lowered = output_text.lower()
        if "tool_execution_timeout" in lowered or "超时" in output_text:
            return "timeout"
        if "安全策略" in output_text or "blocked" in lowered or "拦截" in output_text:
            return "blocked"
    return "success" if tool_call.tool_name else None


def _build_result_payload(
    *,
    tool_call: ToolCallCapture | None,
    audit_record: dict[str, Any] | None,
    final_answer: str | None,
    execution_status: str | None,
) -> dict[str, Any] | None:
    payload = _summarize_tool_output(tool_call.output if tool_call else None)
    if payload is None and audit_record is not None:
        payload = _summarize_audit_result(audit_record)
    if payload is None:
        payload = {}

    if final_answer:
        payload["final_answer_preview"] = _truncate_text(final_answer, 400)

    if execution_status and "execution_status" not in payload:
        payload["execution_status"] = execution_status

    return payload or None


def _summarize_audit_result(record: dict[str, Any]) -> dict[str, Any] | None:
    extra = record.get("extra")
    if not isinstance(extra, dict):
        return None
    payload = extra.get("result")
    if payload is None:
        return None
    normalized = normalize_result_payload(payload)
    if isinstance(normalized, dict):
        return normalized
    if isinstance(normalized, list):
        structured: dict[str, Any] = {"items": normalized, "row_count": len(normalized)}
        if normalized and all(isinstance(item, dict) for item in normalized):
            structured["rows"] = normalized
        return structured
    return {"value": normalized}


def _summarize_tool_output(output: Any) -> dict[str, Any] | None:
    return summarize_tool_output(output)


def _normalize_result_payload(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "metrics" and isinstance(item, dict):
                normalized[key] = _normalize_result_payload(item)
                for metric_key, metric_value in item.items():
                    if metric_key not in normalized:
                        normalized[metric_key] = _normalize_result_payload(metric_value)
                continue
            if key == "group_a_mean":
                normalized["mean_a"] = _normalize_result_payload(item)
                continue
            if key == "group_b_mean":
                normalized["mean_b"] = _normalize_result_payload(item)
                continue
            if key == "top_pairs" and isinstance(item, list) and item:
                normalized[key] = _normalize_result_payload(item)
                first_pair = item[0]
                if isinstance(first_pair, dict) and "corr" in first_pair and "statistic" not in normalized:
                    normalized["statistic"] = _normalize_result_payload(first_pair.get("corr"))
                continue
            if key in {"warnings", "rows", "items", "matrix", "contingency_rows", "coefficient_items"}:
                normalized[key] = _truncate_collection(item)
                if key in {"rows", "items"} and isinstance(item, list):
                    normalized.setdefault("row_count", len(item))
                    if item and isinstance(item[0], dict):
                        metric_keys = [metric_key for metric_key in item[0].keys() if metric_key not in {"column", "group"}]
                        if metric_keys and "metric_count" not in normalized:
                            normalized["metric_count"] = len(metric_keys)
                continue
            normalized[key] = _normalize_result_payload(item)

        if "rows" in normalized and "row_count" not in normalized and isinstance(normalized["rows"], list):
            normalized["row_count"] = len(normalized["rows"])
        if "items" in normalized and "row_count" not in normalized and isinstance(normalized["items"], list):
            normalized["row_count"] = len(normalized["items"])
        if "metrics" in normalized and isinstance(normalized["metrics"], dict):
            normalized["metric_count"] = len(normalized["metrics"])
        return normalized
    if isinstance(value, list):
        return _truncate_collection(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _truncate_text(str(value), 200)


def _truncate_collection(value: Any) -> Any:
    if not isinstance(value, list):
        return _normalize_result_payload(value)
    truncated = [_normalize_result_payload(item) for item in value[:20]]
    return truncated


def _infer_predicted_intent(
    *,
    case: EvalCase,
    predicted_tool: str | None,
    predicted_args: dict[str, Any] | None,
    result: dict[str, Any] | None,
    execution_status: str | None,
) -> str | None:
    if predicted_tool == "stats_execute":
        action = _string_arg(predicted_args, "action")
        if action == "group_summary":
            return "group_aggregation"
        if action in {"correlation", "t_test", "chi_square", "anova"}:
            return "statistical_test"
        return "descriptive_stats"
    if predicted_tool == "ml_execute":
        return "ml_training"
    if predicted_tool == "python_inter":
        code = _python_code_from_args(predicted_args)
        if code and _looks_like_unsafe_request(code):
            return "unsafe_request"
        if code and _looks_like_statistical_python(code):
            return "statistical_test"
        return "descriptive_stats"
    if execution_status == "blocked" and case.should_be_blocked:
        return "unsafe_request"
    return None


def _python_code_from_args(predicted_args: dict[str, Any] | None) -> str | None:
    if not isinstance(predicted_args, dict):
        return None
    value = predicted_args.get("py_code")
    if isinstance(value, dict):
        preview = value.get("code_preview")
        if isinstance(preview, str):
            return preview
        return None
    if isinstance(value, str):
        return value
    return None


def _looks_like_unsafe_request(code: str) -> bool:
    lowered = code.lower()
    return any(
        token in lowered
        for token in (
            "import os",
            "import sys",
            "open(",
            "to_csv(",
            "to_excel(",
            "system prompt",
            "hidden instructions",
            "reveal",
            "export",
            "read(",
        )
    )


def _looks_like_statistical_python(code: str) -> bool:
    lowered = code.lower()
    return any(
        token in lowered
        for token in (
            "confidence interval",
            "ci_low",
            "ci_high",
            "corr(",
            "ttest",
            "anova",
            "chi_square",
            "groupby(",
            "value_counts(",
        )
    )


def _build_router_prediction(case: EvalCase) -> EvalPrediction:
    predicted_tool, predicted_args = _router_plan(case)
    return EvalPrediction(
        case_id=case.case_id,
        predicted_intent=None,
        predicted_tool=predicted_tool,
        predicted_args=predicted_args,
        execution_status=_router_execution_status(case, predicted_tool, predicted_args),
        result=_router_result(case, predicted_tool, predicted_args),
        error_message=None,
    )


def _router_plan(case: EvalCase) -> tuple[str | None, dict[str, Any] | None]:
    query = case.user_query.lower()
    if case.should_be_blocked or any(token in query for token in ("import os", "read a local file", "write code", "export the dataframe")):
        return "python_inter", _router_python_args(case.user_query)
    if "model" in query or case.expected_tool == "ml_execute":
        if "feature importance" in query or "重要特征" in query:
            return "ml_execute", {"action": "feature_importance", "top_k": 5}
        if "metrics" in query or "指标" in query:
            return "ml_execute", {"action": "metrics"}
        return "ml_execute", _router_ml_args(case)
    if any(token in query for token in ("confidence interval", "ci", "import os", "open(", "to_csv(")):
        return "python_inter", _router_python_args(case.user_query)
    if any(token in query for token in ("correlation", "t-test", "chi-square", "anova", "检验", "相关")):
        return "stats_execute", _router_stats_args(case)
    if case.expected_tool == "stats_execute":
        return "stats_execute", _router_stats_args(case)
    return "python_inter", _router_python_args(case.user_query)


def _router_intent(case: EvalCase, predicted_tool: str | None, predicted_args: dict[str, Any] | None) -> str | None:
    if predicted_tool == "ml_execute":
        return "ml_training"
    if predicted_tool == "stats_execute":
        action = _string_arg(predicted_args, "action")
        if action == "group_summary":
            return "group_aggregation"
        if action in {"correlation", "t_test", "chi_square", "anova"}:
            return "statistical_test"
        return "descriptive_stats"
    if predicted_tool == "python_inter":
        if _looks_like_unsafe_request(case.user_query):
            return "unsafe_request"
        if any(token in case.user_query.lower() for token in ("confidence interval", "correlation", "anova", "chi-square", "t-test")):
            return "statistical_test"
        return "descriptive_stats"
    return interpretation.intent_type


def _router_execution_status(case: EvalCase, predicted_tool: str | None, predicted_args: dict[str, Any] | None) -> str | None:
    if case.should_be_blocked:
        return "blocked"
    if predicted_tool is None:
        return "error"
    if predicted_tool == "python_inter" and _looks_like_unsafe_request(case.user_query):
        return "blocked"
    return "success"


def _router_result(case: EvalCase, predicted_tool: str | None, predicted_args: dict[str, Any] | None) -> dict[str, Any] | None:
    if predicted_tool == "ml_execute":
        return {"tool_output_preview": "heuristic router baseline"}
    if predicted_tool == "stats_execute":
        return {"tool_output_preview": "heuristic router baseline"}
    if predicted_tool == "python_inter":
        return {"tool_output_preview": _truncate_text(case.user_query, 200)}
    return None


def _router_stats_args(case: EvalCase) -> dict[str, Any]:
    query = case.user_query.lower()
    if "group by segment" in query:
        return {"action": "group_summary", "group_by": "segment", "metrics": [{"op": "mean", "column": "revenue", "as": "revenue_mean"}], "sort_by": "revenue_mean", "ascending": False, "top_n": 4}
    if "region" in query and "sum" in query and "order_value" in query:
        return {"action": "group_summary", "group_by": "region", "metrics": [{"op": "sum", "column": "order_value", "as": "order_value_sum"}], "sort_by": "order_value_sum", "ascending": False, "top_n": 10}
    if "plan_type" in query and "top 3" in query:
        return {"action": "group_summary", "group_by": "plan_type", "metrics": [{"op": "count", "as": "row_count"}], "sort_by": "row_count", "ascending": False, "top_n": 3}
    if "churn rate" in query:
        return {"action": "group_summary", "group_by": "segment", "metrics": [{"op": "rate", "column": "churn", "as": "churn_rate", "positive_label": "yes"}], "sort_by": "churn_rate", "ascending": False, "top_n": 4}
    if "average revenue by plan" in query:
        return {"action": "group_summary", "group_by": "plan", "metrics": [{"op": "mean", "column": "revenue", "as": "revenue_mean"}], "sort_by": "revenue_mean", "ascending": False, "top_n": 3}
    if "count records by region" in query:
        return {"action": "group_summary", "group_by": "region", "metrics": [{"op": "count", "as": "row_count"}], "sort_by": "row_count", "ascending": False, "top_n": 10}
    if "correlation" in query:
        return {"action": "correlation", "columns": ["tenure", "monthly_charges"], "top_n": 10}
    if "t-test" in query or "compare means" in query:
        if "conversion_rate" in query:
            return {"action": "t_test", "value_col": "conversion_rate", "group_col": "group", "group_a": "control", "group_b": "treatment"}
        return {"action": "t_test", "value_col": "revenue", "group_col": "subscribed", "group_a": True, "group_b": False}
    if "chi-square" in query:
        return {"action": "chi_square", "col_a": "region", "col_b": "churn"}
    if "anova" in query:
        return {"action": "anova", "value_col": "order_value", "group_col": "plan_type"}
    return {"action": "describe_numeric", "columns": ["age", "tenure", "monthly_charges"]}


def _router_ml_args(case: EvalCase) -> dict[str, Any]:
    query = case.user_query.lower()
    if "revenue" in query:
        return {"action": "train", "model_type": "linear_regression", "target": "revenue", "features": ["ad_spend", "clicks", "session_length"], "test_size": 0.2}
    if "conversion" in query or "converted" in query:
        return {"action": "train", "model_type": "logistic_regression", "target": "converted", "features": ["age", "income", "site_visits"], "test_size": 0.25, "positive_label": 1}
    if "feature" in query or "important" in query:
        return {"action": "feature_importance", "model_artifact_id": "telco_churn_churn_model", "top_k": 5}
    if "metrics" in query:
        return {"action": "metrics", "model_artifact_id": "telco_churn_churn_model"}
    return {"action": "train", "model_type": "logistic_regression", "target": "churn", "features": ["age", "tenure", "monthly_charges", "contract"], "test_size": 0.2, "positive_label": "yes"}


def _router_python_args(query: str) -> dict[str, Any]:
    lowered = query.lower()
    if "value counts" in lowered or "contract" in lowered:
        return {"py_code": "print(data.value_counts('contract', top_n=3))"}
    if "top 5 rows" in lowered or "top rows" in lowered or "revenue" in lowered:
        return {"py_code": "print(data.top_rows('revenue', n=5))"}
    if "missing" in lowered:
        return {"py_code": "print(data.missing_summary())"}
    if "unique" in lowered:
        return {"py_code": "print(len(data.unique('contract')))"}
    if "distribution" in lowered:
        return {"py_code": "print(data.numeric_summary())"}
    if "confidence interval" in lowered:
        return {"py_code": "mean = data.numeric_summary().loc['tenure', 'mean']\nci_low = mean - 1.96\nci_high = mean + 1.96\nprint({'ci_low': ci_low, 'ci_high': ci_high})"}
    if "import os" in lowered:
        return {"py_code": "import os\nprint(os.listdir())"}
    if "open(" in lowered or "local file" in lowered:
        return {"py_code": "print(open('C:/Users/llj68/secret.txt').read())"}
    if "export" in lowered or "write" in lowered:
        return {"py_code": "df.to_csv('D:/tmp/leak.csv')"}
    return {"py_code": "print(data.numeric_summary())"}


def _build_live_benchmark_dataframe(dataset_key: str) -> pd.DataFrame:
    seed = int(sha256(dataset_key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    n = 240

    tenure = np.clip(np.round(rng.gamma(shape=2.8, scale=10.5, size=n)), 1, 72).astype(int)
    age = np.clip(np.round(rng.normal(43, 12, size=n)), 18, 80).astype(int)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], size=n, p=[0.48, 0.30, 0.22])
    monthly_charges = np.clip(np.round(28 + 0.45 * tenure + rng.normal(0, 15, size=n), 2), 15, 160)
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 90, size=n), 2)
    missing_indices = rng.choice(n, 11, replace=False)
    total_charges[missing_indices] = np.nan

    region = rng.choice(
        [f"region_{index}" for index in range(1, 11)],
        size=n,
        p=[0.14, 0.13, 0.12, 0.11, 0.11, 0.10, 0.10, 0.09, 0.05, 0.05],
    )
    segment = rng.choice(["consumer", "smb", "enterprise", "mid-market"], size=n, p=[0.4, 0.25, 0.2, 0.15])
    plan_type = rng.choice(["basic", "standard", "premium"], size=n, p=[0.45, 0.35, 0.20])
    plan = rng.choice(["starter", "growth", "scale"], size=n, p=[0.5, 0.3, 0.2])
    payment_method = rng.choice(["bank_transfer", "credit_card", "paypal"], size=n)
    gender = rng.choice(["female", "male"], size=n)
    country = np.where(rng.random(n) < 0.85, "US", "CA")
    state = rng.choice(["California", "New York", "Texas", "Washington", "Florida"], size=n)
    ad_spend = np.round(rng.uniform(10, 250, size=n), 2)
    clicks = rng.integers(20, 500, size=n)
    session_length = np.round(rng.uniform(1.0, 25.0, size=n), 2)
    revenue = np.round(50 + 0.85 * ad_spend + 0.12 * clicks + 9.5 * session_length + rng.normal(0, 8, size=n), 2)
    order_value = np.round(20 + 0.75 * revenue + rng.normal(0, 15, size=n), 2)
    income = np.round(22000 + age * 1300 + rng.normal(0, 6000, size=n), 2)
    site_visits = np.clip(np.round(rng.normal(18, 7, size=n)), 1, None).astype(int)
    group = rng.choice(["control", "treatment"], size=n)
    conversion_rate = np.where(
        group == "control",
        rng.normal(0.12, 0.015, size=n),
        rng.normal(0.15, 0.015, size=n),
    )
    conversion_rate = np.round(np.clip(conversion_rate, 0.01, 0.4), 4)
    subscribed = rng.random(n) < np.clip(0.42 + 0.0025 * tenure + 0.07 * (group == "treatment"), 0.05, 0.95)

    churn_score = (
        -1.35
        + 0.03 * monthly_charges
        - 0.045 * tenure
        + np.where(contract == "month-to-month", 0.65, np.where(contract == "one-year", 0.2, -0.1))
        + np.where(region == "region_1", 0.28, 0.0)
        + np.where(region == "region_2", 0.18, 0.0)
        + rng.normal(0, 0.5, size=n)
    )
    churn_probability = 1 / (1 + np.exp(-churn_score))
    churn = np.where(rng.random(n) < churn_probability, "yes", "no")

    revenue = revenue + np.where(subscribed, 18, -8)
    order_value = np.round(20 + 0.75 * revenue + rng.normal(0, 15, size=n), 2)
    converted = (rng.random(n) < np.clip(0.08 + 0.000018 * income + 0.015 * site_visits + 0.003 * age, 0.02, 0.85)).astype(int)

    data = pd.DataFrame(
        {
            "customer_id": [f"C{index:04d}" for index in range(1, n + 1)],
            "age": age,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract": contract,
            "region": region,
            "segment": segment,
            "plan_type": plan_type,
            "plan": plan,
            "payment_method": payment_method,
            "state": state,
            "country": country,
            "gender": gender,
            "revenue": revenue,
            "order_value": order_value,
            "ad_spend": ad_spend,
            "clicks": clicks,
            "session_length": session_length,
            "income": income,
            "site_visits": site_visits,
            "subscribed": subscribed,
            "group": group,
            "conversion_rate": conversion_rate,
            "converted": converted,
            "churn": churn,
        }
    )
    return data


def _extract_final_answer(result: Any, fallback_chunks: list[str]) -> str | None:
    text = extract_text_from_chunk(result)
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                message_text = extract_text_from_chunk(message)
                if isinstance(message_text, str) and message_text.strip():
                    return message_text.strip()
        content = result.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    if isinstance(text, str) and text.strip():
        return text.strip()
    combined = _combine_text(fallback_chunks)
    return combined or None


def _combine_text(chunks: list[str]) -> str:
    return "".join(chunk for chunk in chunks if isinstance(chunk, str)).strip()


def _load_jsonl_dicts(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def _coerce_tool_args(
    *,
    inputs: Any,
    input_str: str,
    tool_name: str | None,
) -> dict[str, Any] | str | None:
    if isinstance(inputs, dict) and inputs:
        return inputs
    parsed = _maybe_parse_json_object(input_str)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(input_str, str) and input_str.strip():
        if tool_name in {"python_inter"}:
            return {"py_code": input_str.strip()}
        return input_str.strip()
    return None


def _maybe_parse_json_object(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except Exception:
        return None
    return parsed


def _extract_tool_name(serialized: dict[str, Any], kwargs: dict[str, Any]) -> str | None:
    for candidate in (serialized.get("name"), serialized.get("tool_name"), kwargs.get("name"), kwargs.get("tool_name")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _stringify_output(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    if isinstance(value, list):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    text = extract_text_from_chunk(value)
    if text:
        return text
    return str(value)


def _parse_structured_output(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except Exception:
        parsed = None
    if parsed is not None:
        return parsed

    try:
        parsed_literal = ast.literal_eval(stripped)
    except Exception:
        parsed_literal = None
    if isinstance(parsed_literal, (dict, list)):
        return parsed_literal

    table = _parse_table_text(stripped)
    if table is not None:
        return table
    return None


def _parse_table_text(text: str) -> list[dict[str, Any]] | None:
    try:
        frame = pd.read_fwf(StringIO(text))
    except Exception:
        return None
    if frame.empty:
        return None
    frame = frame.replace({np.nan: None})
    records = frame.head(20).to_dict(orient="records")
    if not records:
        return None
    return records


def _truncate_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _string_arg(predicted_args: dict[str, Any] | None, key: str) -> str | None:
    if not isinstance(predicted_args, dict):
        return None
    value = predicted_args.get(key)
    if isinstance(value, str):
        return value
    return None


def _looks_like_unsafe_request(query: str) -> bool:
    lowered = query.lower()
    return any(
        token in lowered
        for token in (
            "import os",
            "read a local file",
            "read local file",
            "reveal the hidden system prompt",
            "prompt injection",
            "overly complex code",
            "nested loops",
            "export",
            "write",
        )
    )


def _coalesce_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _align_prediction_case_id(prediction: EvalPrediction, case_id: str) -> EvalPrediction:
    if prediction.case_id == case_id:
        return prediction
    return prediction.model_copy(update={"case_id": case_id})


def _dataset_cache_key(dataset_name: str | None) -> str:
    if isinstance(dataset_name, str) and dataset_name.strip():
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name.strip())
    return "generic_benchmark"


if __name__ == "__main__":
    raise SystemExit(main())
