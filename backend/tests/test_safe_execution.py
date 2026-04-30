from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.tools import (
    ReadOnlyDataFrameProxy,
    SafeExecutionError,
    SafePythonExecutor,
    _build_helper_api,
)


def _build_env():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6], "cat": ["a", "b", "a"]})
    data, viz, stats, profile, ml = _build_helper_api(df)
    return {
        "df": ReadOnlyDataFrameProxy(df),
        "data": data,
        "viz": viz,
        "stats": stats,
        "profile": profile,
        "ml": ml,
    }


@pytest.mark.parametrize(
    ("code", "needle"),
    [
        ("import os", "import"),
        ("print(open('x.txt', 'w'))", "open"),
        ("print(eval('1+1'))", "eval"),
        ("exec('print(1)')", "exec"),
        ("print(data._df)", "敏感属性"),
        ("print(df.to_excel('x.xlsx'))", "to_excel"),
        ("writer = df['x'].to_csv\nwriter('x.csv')", "to_csv"),
    ],
)
def test_python_guardrails_block_dangerous_code(tmp_path: Path, code: str, needle: str):
    executor = SafePythonExecutor(image_dir=tmp_path)
    with pytest.raises(SafeExecutionError) as exc_info:
        executor.safe_execute_python(code, _build_env())
    assert needle in str(exc_info.value)


def test_python_guardrails_do_not_write_via_method_alias(tmp_path: Path):
    executor = SafePythonExecutor(image_dir=tmp_path)
    output_path = tmp_path / "leak.csv"
    code = f"writer = df['x'].to_csv\nwriter(r'{output_path.as_posix()}')"

    with pytest.raises(SafeExecutionError):
        executor.safe_execute_python(code, _build_env())

    assert not output_path.exists()


def test_read_only_dataframe_proxy_still_supports_filtering(tmp_path: Path):
    executor = SafePythonExecutor(image_dir=tmp_path)
    result = executor.safe_execute_python("print(df[df['x'] > 1].shape)", _build_env())
    assert "(2, 3)" in result


def test_plot_code_cannot_save_arbitrary_paths(tmp_path: Path):
    executor = SafePythonExecutor(image_dir=tmp_path)
    code = "fig = viz.hist('x')\nfig.savefig('D:/tmp/escape.png')"
    with pytest.raises(SafeExecutionError) as exc_info:
        executor.safe_execute_plot(code, _build_env(), "fig")
    assert "savefig" in str(exc_info.value)


def test_plot_saves_only_to_allowed_image_dir(tmp_path: Path):
    executor = SafePythonExecutor(image_dir=tmp_path)
    result = executor.safe_execute_plot("fig = viz.hist('x')", _build_env(), "fig")
    assert "图表已生成" in result
    images = list(tmp_path.glob("*.png"))
    assert len(images) == 1
    assert images[0].parent.resolve() == tmp_path.resolve()


def test_execution_timeout_blocks_infinite_loop(tmp_path: Path):
    executor = SafePythonExecutor(image_dir=tmp_path)
    with pytest.raises(SafeExecutionError) as exc_info:
        executor.safe_execute_python("while True:\n    pass", _build_env())
    assert "超时" in str(exc_info.value)
