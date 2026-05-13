from __future__ import annotations

from collections import Counter
from math import isclose
from numbers import Real
import hashlib
import re
from typing import Any

from .schema import EvalCase, EvalPrediction, EvalScores

_DYNAMIC_ARGUMENT_KEYS = {
    "artifact_id",
    "chart_id",
    "dataset_id",
    "job_id",
    "model_artifact_id",
    "model_id",
    "run_id",
    "source_model_artifact_id",
}
_CODE_ARGUMENT_KEYS = {"code", "py_code"}


def intent_accuracy(cases: list[EvalCase], predictions: list[EvalPrediction]) -> float | None:
    return _label_accuracy(
        cases,
        predictions,
        expected_getter=lambda case: case.expected_intent,
        accepted_getter=lambda case: case.accepted_intents,
        predicted_getter=lambda prediction: prediction.predicted_intent,
    )


def tool_accuracy(cases: list[EvalCase], predictions: list[EvalPrediction]) -> float | None:
    return _label_accuracy(
        cases,
        predictions,
        expected_getter=lambda case: case.expected_tool,
        accepted_getter=lambda case: case.accepted_tools,
        predicted_getter=lambda prediction: prediction.predicted_tool,
    )


def argument_f1(cases: list[EvalCase], predictions: list[EvalPrediction]) -> float | None:
    expected_total = 0
    predicted_total = 0
    matched_total = 0

    prediction_map = _prediction_map(predictions)
    for case in cases:
        if case.expected_args is None:
            continue
        prediction = prediction_map.get(case.case_id)
        expected_tokens = _argument_tokens(_normalize_args_for_scoring(case.expected_args))
        predicted_tokens = _argument_tokens(
            _normalize_args_for_scoring(prediction.predicted_args if prediction else None)
        )
        expected_total += sum(expected_tokens.values())
        predicted_total += sum(predicted_tokens.values())
        matched_total += sum((expected_tokens & predicted_tokens).values())

    return _f1_from_counts(matched_total, predicted_total, expected_total)


def execution_success_rate(cases: list[EvalCase], predictions: list[EvalPrediction]) -> float | None:
    prediction_map = _prediction_map(predictions)
    correct = 0
    total = 0

    for case in cases:
        if case.should_execute_successfully is None:
            continue
        total += 1
        prediction = prediction_map.get(case.case_id)
        if prediction is None:
            continue
        status = prediction.execution_status if prediction else None
        if case.should_execute_successfully:
            if status == "success":
                correct += 1
        elif status != "success":
            correct += 1

    return _ratio(correct, total)


def blocked_request_accuracy(cases: list[EvalCase], predictions: list[EvalPrediction]) -> float | None:
    prediction_map = _prediction_map(predictions)
    correct = 0
    total = 0

    for case in cases:
        if case.should_be_blocked is None:
            continue
        total += 1
        prediction = prediction_map.get(case.case_id)
        if prediction is None:
            continue
        status = prediction.execution_status if prediction else None
        if case.should_be_blocked:
            if status == "blocked":
                correct += 1
        elif status != "blocked":
            correct += 1

    return _ratio(correct, total)


def numerical_accuracy(cases: list[EvalCase], predictions: list[EvalPrediction]) -> float | None:
    prediction_map = _prediction_map(predictions)
    correct = 0
    total = 0

    for case in cases:
        if case.expected_result is None or case.stable_numerical_result is not True:
            continue
        prediction = prediction_map.get(case.case_id)
        expected_fields = _flatten_numeric_fields(case.expected_result)
        predicted_fields = _flatten_numeric_fields(prediction.result if prediction else None)
        predicted_index = _index_numeric_fields(predicted_fields)
        for field_path, expected_value in expected_fields.items():
            predicted_value = predicted_fields.get(field_path)
            if predicted_value is None:
                predicted_value = _match_numeric_field_by_terminal(field_path, predicted_index)
            if predicted_value is None:
                continue
            total += 1
            if isclose(predicted_value, expected_value, rel_tol=1e-3, abs_tol=1e-6):
                correct += 1

    return _ratio(correct, total)


def score_predictions(cases: list[EvalCase], predictions: list[EvalPrediction]) -> EvalScores:
    return EvalScores(
        num_cases=len(cases),
        intent_accuracy=intent_accuracy(cases, predictions),
        tool_accuracy=tool_accuracy(cases, predictions),
        argument_f1=argument_f1(cases, predictions),
        execution_success_rate=execution_success_rate(cases, predictions),
        blocked_request_accuracy=blocked_request_accuracy(cases, predictions),
        numerical_accuracy=numerical_accuracy(cases, predictions),
    )


def _ratio(correct: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(correct / total, 6)


def _f1_from_counts(matched: int, predicted_total: int, expected_total: int) -> float | None:
    if expected_total <= 0:
        return None
    if predicted_total <= 0:
        return 0.0
    precision = matched / predicted_total
    recall = matched / expected_total
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 6)


def _ratio_metric(
    cases: list[EvalCase],
    predictions: list[EvalPrediction],
    *,
    expected_getter,
    predicted_getter,
) -> float | None:
    return _label_accuracy(
        cases,
        predictions,
        expected_getter=expected_getter,
        accepted_getter=lambda _: None,
        predicted_getter=predicted_getter,
    )


def _label_accuracy(
    cases: list[EvalCase],
    predictions: list[EvalPrediction],
    *,
    expected_getter,
    accepted_getter,
    predicted_getter,
) -> float | None:
    prediction_map = _prediction_map(predictions)
    correct = 0
    total = 0

    for case in cases:
        expected = _normalize_label(expected_getter(case))
        accepted = _normalize_label_list(accepted_getter(case))
        if expected is None and not accepted:
            continue
        total += 1
        prediction = prediction_map.get(case.case_id)
        predicted = _normalize_label(predicted_getter(prediction) if prediction else None)
        if predicted is None:
            continue
        if expected is not None and predicted == expected:
            correct += 1
            continue
        if predicted in accepted:
            correct += 1

    return _ratio(correct, total)


def _prediction_map(predictions: list[EvalPrediction]) -> dict[str, EvalPrediction]:
    result: dict[str, EvalPrediction] = {}
    for prediction in predictions:
        if prediction.case_id in result:
            raise ValueError(f"Duplicate prediction case_id: {prediction.case_id}")
        result[prediction.case_id] = prediction
    return result


def _argument_tokens(value: Any) -> Counter[tuple[str, Any]]:
    tokens: Counter[tuple[str, Any]] = Counter()

    def walk(current: Any, path: str) -> None:
        if isinstance(current, dict):
            if not current:
                tokens[(path or "<root>", ("dict", ()))] += 1
                return
            for key in sorted(current):
                child_path = f"{path}.{key}" if path else str(key)
                walk(current[key], child_path)
            return
        if isinstance(current, list):
            if not current:
                tokens[(path or "<root>", ("list", ()))] += 1
                return
            for item in current:
                child_path = f"{path}[]" if path else "[]"
                walk(item, child_path)
            return
        tokens[(path or "<root>", _freeze_scalar(current))] += 1

    if value is not None:
        walk(value, "")
    return tokens


def _normalize_args_for_scoring(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return _normalize_args_dict(value)
    if isinstance(value, list):
        items = [_normalize_args_for_scoring(item) for item in value]
        return [item for item in items if item is not None]
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _normalize_args_dict(value: dict[str, Any]) -> dict[str, Any] | None:
    normalized: dict[str, Any] = {}
    for key in sorted(value):
        if key in _DYNAMIC_ARGUMENT_KEYS:
            continue
        child = value[key]
        if key in _CODE_ARGUMENT_KEYS:
            code_value = _normalize_code_value(child)
            if code_value is not None:
                normalized[key] = code_value
            continue
        normalized_child = _normalize_args_for_scoring(child)
        if normalized_child is not None:
            normalized[key] = normalized_child
    return normalized or None


def _normalize_code_value(value: Any) -> dict[str, str] | None:
    if isinstance(value, dict):
        if "code_sha256" in value or "code_preview" in value:
            normalized: dict[str, str] = {}
            code_sha256 = value.get("code_sha256")
            code_preview = value.get("code_preview")
            if isinstance(code_sha256, str) and code_sha256.strip():
                normalized["code_sha256"] = code_sha256.strip()
            if isinstance(code_preview, str) and code_preview.strip():
                normalized["code_preview"] = code_preview
            return normalized or None
        if "py_code" in value or "code" in value:
            for key in ("py_code", "code"):
                code = value.get(key)
                if isinstance(code, str) and code.strip():
                    return _summarize_code(code.strip())
        return _normalize_args_dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return _summarize_code(stripped)
    return None


def _summarize_code(code: str) -> dict[str, str]:
    return {
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "code_preview": code[:200],
    }


def _freeze_scalar(value: Any) -> tuple[str, Any]:
    if value is None:
        return ("none", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int) and not isinstance(value, bool):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value.decode("utf-8", errors="replace"))
    return ("repr", repr(value))


def _index_numeric_fields(fields: dict[str, float]) -> dict[str, list[float]]:
    index: dict[str, list[float]] = {}
    for path, value in fields.items():
        terminal_key = _terminal_numeric_key(path)
        index.setdefault(terminal_key, []).append(value)
    return index


def _match_numeric_field_by_terminal(path: str, index: dict[str, list[float]]) -> float | None:
    terminal_key = _terminal_numeric_key(path)
    candidates = index.get(terminal_key)
    if not candidates:
        return None
    unique_candidates = list(dict.fromkeys(candidates))
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    return None


def _terminal_numeric_key(path: str) -> str:
    tokens = [token for token in re.split(r"[.\[]", path) if token]
    if not tokens:
        return path
    terminal = tokens[-1]
    return terminal.rstrip("]")


def _normalize_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped.lower() if stripped else None


def _normalize_label_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        normalized = _normalize_label(values)
        return [normalized] if normalized else []
    normalized_values: list[str] = []
    for value in values:
        normalized = _normalize_label(value)
        if normalized and normalized not in normalized_values:
            normalized_values.append(normalized)
    return normalized_values


def _flatten_numeric_fields(value: Any, *, path: str = "") -> dict[str, float]:
    fields: dict[str, float] = {}
    if value is None:
        return fields
    if isinstance(value, dict):
        for key in sorted(value):
            child_path = f"{path}.{key}" if path else str(key)
            fields.update(_flatten_numeric_fields(value[key], path=child_path))
        return fields
    if isinstance(value, list):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            fields.update(_flatten_numeric_fields(item, path=child_path))
        return fields
    if isinstance(value, Real) and not isinstance(value, bool):
        fields[path or "<root>"] = float(value)
    return fields
