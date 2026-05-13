from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from .metrics import score_predictions
from .schema import EvalCase, EvalPrediction, EvalScores

TModel = TypeVar("TModel", EvalCase, EvalPrediction)


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    return _load_jsonl_models(path, EvalCase, label="eval case")


def load_predictions(path: str | Path) -> list[EvalPrediction]:
    return _load_jsonl_models(path, EvalPrediction, label="prediction")


def score_prediction_files(cases_path: str | Path, predictions_path: str | Path) -> EvalScores:
    cases = load_eval_cases(cases_path)
    predictions = load_predictions(predictions_path)
    return score_predictions(cases, predictions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline evaluation runner for the data agent baseline.")
    parser.add_argument(
        "--cases",
        default="evals/baseline_cases.jsonl",
        help="Path to the JSONL file containing EvalCase records.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to the JSONL file containing EvalPrediction records.",
    )
    args = parser.parse_args(argv)

    try:
        scores = score_prediction_files(args.cases, args.predictions)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(scores.model_dump(), ensure_ascii=False, indent=2))
    return 0


def _load_jsonl_models(path: str | Path, model_type: type[TModel], *, label: str) -> list[TModel]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label.title()} file not found: {file_path}")

    records: list[TModel] = []
    with file_path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(f"Empty line at {line_number} in {file_path}")
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {file_path}: {exc.msg}") from exc
            try:
                record = model_type.model_validate(parsed)
            except ValidationError as exc:
                raise ValueError(f"Invalid {label} on line {line_number} in {file_path}: {exc}") from exc
            records.append(record)
    return records


if __name__ == "__main__":
    raise SystemExit(main())
