# Offline Evaluation Baseline

This directory contains lightweight offline evaluation assets for the Python data agent.

## Files

- `baseline_cases.jsonl`: canonical baseline evaluation cases

## Case schema

Each JSONL row is an `EvalCase` with:

- `case_id`
- `category`
- `user_query`
- `dataset_name`
- `expected_intent`
- `expected_tool`
- `expected_args`
- `expected_result`
- `should_execute_successfully`
- `should_be_blocked`
- `notes`

Optional fields may be set to `null` when they are not relevant for a case.

## Runner

Score a prediction file with:

```bash
python -m src.eval.runner --cases evals/baseline_cases.jsonl --predictions path/to/predictions.jsonl
```

## Prediction Files

### 1. Smoke Test

`evals/sample_predictions.jsonl` is only a small smoke test file. It is useful for checking the runner pipeline, but it is **not** a real agent benchmark.

### 2. Blank Template

Create a blank template from the baseline cases:

```bash
python -m src.eval.capture_baseline --mode template --cases evals/baseline_cases.jsonl --out evals/baseline_predictions.template.jsonl
```

Create a prediction file from audit logs when audit records include `extra.case_id`:

```bash
python -m src.eval.capture_baseline --mode audit --audit-log audit_logs/audit.jsonl --out evals/baseline_predictions.from_audit.jsonl
```

### 3. Heuristic Router Baseline

`--mode router` is a key-free heuristic/offline baseline. It does **not** call DeepSeek and it is **not** the live benchmark.

```bash
python -m src.eval.capture_baseline --mode router --cases evals/baseline_cases.jsonl --out evals/baseline_predictions.router.jsonl
```

### 4. Full Live Agent Benchmark

`--mode live` is the real benchmark path. It calls the existing DeepSeek / LangChain agent, lets the LLM choose the tool, executes the tool, and captures an `EvalPrediction` JSONL file.

`DEEPSEEK_API_KEY` is required for live capture. If it is missing, the command fails immediately and does not fall back to any heuristic mode.

Recommended first pass:

```bash
python -m src.eval.capture_baseline --mode live --cases evals/baseline_cases.jsonl --out evals/baseline_predictions.live.5.jsonl --limit 5
```

Then score it with:

```bash
python -m src.eval.runner --cases evals/baseline_cases.jsonl --predictions evals/baseline_predictions.live.5.jsonl
```

Sample predictions can be scored directly:

```bash
python -m src.eval.runner --cases evals/baseline_cases.jsonl --predictions evals/sample_predictions.jsonl
```

The predictions file must be JSONL with `EvalPrediction` rows. The runner:

- loads both files strictly
- aligns predictions by `case_id`
- treats missing predictions as incorrect where a metric is applicable
- prints a JSON summary to stdout

## Notes

- JSONL parsing is strict by design.
- Malformed lines raise a `ValueError` with the line number.
- The baseline cases are offline metadata only; they do not execute the agent.
- Unit tests do not call the live LLM.
- Live baseline capture should be run manually only when API keys/config are available.
