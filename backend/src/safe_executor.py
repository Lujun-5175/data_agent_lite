"""
Safe Python executor — AST validation, sandboxed execution, read-only proxies.
Extracted from tools.py for modularity.
"""
from __future__ import annotations

import ast
import contextlib
import logging
import platform
import signal
import sys
import time
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.errors import AppError
from src.settings import SETTINGS

logger = logging.getLogger(__name__)

# ── Context vars ──────────────────────────────────────────────
CURRENT_DATASET_ID: ContextVar[str | None] = ContextVar("current_dataset_id", default=None)

def set_current_dataset_id(dataset_id: str | None) -> None:
    CURRENT_DATASET_ID.set(dataset_id)

def get_current_dataset_id() -> str | None:
    return CURRENT_DATASET_ID.get()

@contextlib.contextmanager
def bind_current_dataset_id(dataset_id: str | None):
    token = CURRENT_DATASET_ID.set(dataset_id)
    try:
        yield
    finally:
        CURRENT_DATASET_ID.reset(token)

# ── Security constants ────────────────────────────────────────
ALLOWED_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "dir": dir, "enumerate": enumerate, "float": float,
    "int": int, "len": len, "list": list, "max": max, "min": min,
    "print": print, "range": range, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip,
}

FORBIDDEN_CALL_NAMES = {
    "__import__", "breakpoint", "compile", "delattr",
    "eval", "exec", "getattr", "globals", "input",
    "locals", "open", "setattr", "vars",
}

FORBIDDEN_METHOD_NAMES = {
    "boxplot", "dump", "dumps", "export", "from_file", "from_url",
    "imsave", "load", "loads", "read", "read_csv", "read_excel",
    "read_feather", "read_fwf", "read_hdf", "read_html", "read_json",
    "read_orc", "read_parquet", "read_pickle", "read_sas", "read_spss",
    "read_sql", "read_stata", "read_table", "read_xml", "save",
    "savefig", "to_excel", "to_clipboard", "to_csv", "to_feather",
    "to_gbq", "to_hdf", "to_html", "to_json", "to_latex", "to_markdown",
    "to_orc", "to_parquet", "to_pickle", "to_sql", "to_stata", "to_xml",
    "tofile", "write", "write_html", "write_image",
}

FORBIDDEN_IDENTIFIERS = {
    "matplotlib", "os", "pathlib",
    "requests", "seaborn", "shutil", "socket", "subprocess", "sys",
}

# ── Exceptions ────────────────────────────────────────────────
class SafeExecutionError(AppError):
    """Raised when code fails static validation or safe execution."""
    def __init__(self, message: str) -> None:
        super().__init__("invalid_python_code", message, 400)

class ToolExecutionTimeoutError(SafeExecutionError):
    """Raised when constrained execution exceeds its adaptive timeout budget."""
    def __init__(self, seconds: float) -> None:
        super().__init__(
            f"Code execution timed out (> {seconds:.1f}s). "
            f"Try narrowing scope, specifying columns, or reducing loops."
        )
        self.code = "tool_execution_timeout"

# ── AST Validator ─────────────────────────────────────────────
class SafeCodeValidator(ast.NodeVisitor):
    def __init__(
        self, *, max_ast_nodes: int | None = None,
        max_loop_nesting: int | None = None,
        max_comprehension_nesting: int | None = None,
        max_call_chain_depth: int | None = None,
    ) -> None:
        self.max_ast_nodes = max_ast_nodes or SETTINGS.safe_exec_max_ast_nodes
        self.max_loop_nesting = max_loop_nesting or SETTINGS.safe_exec_max_loop_nesting
        self.max_comprehension_nesting = max_comprehension_nesting or SETTINGS.safe_exec_max_comprehension_nesting
        self.max_call_chain_depth = max_call_chain_depth or SETTINGS.safe_exec_max_call_chain_depth
        self._ast_node_count = 0
        self._loop_nesting = 0
        self._comprehension_nesting = 0

    def visit(self, node: ast.AST) -> Any:
        self._ast_node_count += 1
        if self._ast_node_count > self.max_ast_nodes:
            raise SafeExecutionError("Code too complex: AST node count exceeds limit.")
        return super().visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        raise SafeExecutionError("Import statements are not allowed in executed code.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        raise SafeExecutionError("From...import statements are not allowed in executed code.")

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id.startswith("__"):
            raise SafeExecutionError(f"Access to forbidden name: {node.id}")
        if node.id in FORBIDDEN_IDENTIFIERS:
            raise SafeExecutionError(f"Access to dangerous identifier: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        self._check_call_chain_depth(node)
        if node.attr.startswith("_"):
            raise SafeExecutionError(f"Access to sensitive attribute: {node.attr}")
        if node.attr in FORBIDDEN_METHOD_NAMES:
            raise SafeExecutionError(f"Access to dangerous function: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        self._check_call_chain_depth(node)
        call_name = self._resolve_call_name(node.func)
        if call_name in FORBIDDEN_CALL_NAMES or call_name in FORBIDDEN_METHOD_NAMES:
            raise SafeExecutionError(f"Call to dangerous function: {call_name}")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        self._visit_loop_node(node)
    def visit_While(self, node: ast.While) -> Any:
        self._visit_loop_node(node)
    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        self._visit_loop_node(node)
    def visit_ListComp(self, node: ast.ListComp) -> Any:
        self._visit_comprehension_node(node)
    def visit_SetComp(self, node: ast.SetComp) -> Any:
        self._visit_comprehension_node(node)
    def visit_DictComp(self, node: ast.DictComp) -> Any:
        self._visit_comprehension_node(node)
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        self._visit_comprehension_node(node)

    def _resolve_call_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _visit_loop_node(self, node: ast.AST) -> Any:
        self._loop_nesting += 1
        try:
            if self._loop_nesting > self.max_loop_nesting:
                raise SafeExecutionError("Code too complex: loop nesting exceeds limit.")
            self.generic_visit(node)
        finally:
            self._loop_nesting -= 1

    def _visit_comprehension_node(self, node: ast.AST) -> Any:
        self._comprehension_nesting += 1
        try:
            if self._comprehension_nesting > self.max_comprehension_nesting:
                raise SafeExecutionError("Code too complex: comprehension nesting exceeds limit.")
            self.generic_visit(node)
        finally:
            self._comprehension_nesting -= 1

    def _check_call_chain_depth(self, node: ast.AST) -> None:
        if self._measure_chain_depth(node) > self.max_call_chain_depth:
            raise SafeExecutionError("Code too complex: call chain too long.")

    def _measure_chain_depth(self, node: ast.AST) -> int:
        if isinstance(node, ast.Call):
            return 1 + self._measure_chain_depth(node.func)
        if isinstance(node, ast.Attribute):
            return 1 + self._measure_chain_depth(node.value)
        return 0

# ── Execution helpers ────────────────────────────────────────
class _StdoutCollector:
    def __init__(self, chunks: list[str]):
        self._chunks = chunks
    def write(self, value: str) -> int:
        self._chunks.append(value)
        return len(value)
    def flush(self) -> None:
        pass

@contextlib.contextmanager
def _execution_timeout(seconds: float):
    if platform.system() == "Windows":
        yield
        return
    def _handler(signum: int, frame: Any) -> None:
        raise ToolExecutionTimeoutError(seconds)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

# ── Read-only proxies ────────────────────────────────────────
class ReadOnlyDataFrameProxy:
    def __init__(self, df: Any) -> None:
        object.__setattr__(self, "_df", df)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise SafeExecutionError(f"Access to forbidden attribute: {name}")
        if name in FORBIDDEN_METHOD_NAMES:
            raise SafeExecutionError(f"Call to dangerous function: {name}")
        attr = getattr(self._df, name)
        if name == "plot":
            raise SafeExecutionError(f"df is currently read-only, attribute not available: {name}")
        if callable(attr):
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                result = attr(*args, **kwargs)
                if isinstance(result, type(self._df)):
                    return ReadOnlyDataFrameProxy(result)
                return result
            return _wrapped
        return attr

    def __getitem__(self, key: Any) -> Any:
        result = self._df[key]
        if isinstance(result, pd.DataFrame):
            return ReadOnlyDataFrameProxy(result)
        if isinstance(result, pd.Series):
            return ReadOnlySeriesProxy(result)
        return result

    def __setattr__(self, name: str, value: Any) -> None:
        raise SafeExecutionError("DataFrame is read-only. Cannot assign values.")
    def __delitem__(self, key: Any) -> None:
        raise SafeExecutionError("DataFrame is read-only. Cannot delete columns.")
    def __setitem__(self, key: Any, value: Any) -> None:
        raise SafeExecutionError("DataFrame is read-only. Cannot modify data.")
    def __repr__(self) -> str:
        return repr(self._df)
    def __str__(self) -> str:
        return str(self._df)

class ReadOnlySeriesProxy:
    def __init__(self, series: Any) -> None:
        object.__setattr__(self, "_series", series)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise SafeExecutionError(f"Access to forbidden attribute: {name}")
        if name in FORBIDDEN_METHOD_NAMES:
            raise SafeExecutionError(f"Call to dangerous function: {name}")
        attr = getattr(self._series, name)
        if callable(attr):
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                result = attr(*args, **kwargs)
                if isinstance(result, type(self._series)):
                    return ReadOnlySeriesProxy(result)
                return result
            return _wrapped
        return attr

    def __getitem__(self, key: Any) -> Any:
        result = self._series[key]
        if isinstance(result, pd.Series):
            return ReadOnlySeriesProxy(result)
        return result

    # Comparison operators — delegate to the underlying Series
    # so that df[df["col"] > 5] works through boolean indexing.
    def __gt__(self, other: Any) -> pd.Series:
        return self._series > other
    def __lt__(self, other: Any) -> pd.Series:
        return self._series < other
    def __ge__(self, other: Any) -> pd.Series:
        return self._series >= other
    def __le__(self, other: Any) -> pd.Series:
        return self._series <= other
    def __eq__(self, other: Any) -> pd.Series:  # type: ignore[override]
        return self._series == other
    def __ne__(self, other: Any) -> pd.Series:  # type: ignore[override]
        return self._series != other

    def __setattr__(self, name: str, value: Any) -> None:
        raise SafeExecutionError("Series is read-only. Cannot assign values.")
    def __setitem__(self, key: Any, value: Any) -> None:
        raise SafeExecutionError("Series is read-only. Cannot modify data.")
    def __repr__(self) -> str:
        return repr(self._series)
    def __str__(self) -> str:
        return str(self._series)

# ── Safe Python Executor (module-level functions) ───────────
def safe_execute_python(py_code: str, env: dict[str, Any], *, df: pd.DataFrame | None = None) -> str:
    compiled = _validate_and_compile(py_code)
    output: list[str] = []
    execution_env = _build_env(env)
    timeout_seconds = _resolve_timeout_seconds(df=df, mode="python")

    try:
        with contextlib.redirect_stdout(_StdoutCollector(output)):
            with _execution_timeout(timeout_seconds):
                exec(compiled, execution_env, execution_env)
    except SafeExecutionError:
        raise
    except Exception as exc:
        logger.warning("Safe Python execution failed: %s", exc)
        return f"Code execution failed: {exc}"

    printed = "".join(output).strip()
    if printed:
        return printed
    return "Code executed successfully, but no output. Use print() to display results."


def _validate_and_compile(py_code: str):
    try:
        tree = ast.parse(py_code, mode="exec")
    except SyntaxError as exc:
        raise SafeExecutionError(f"Python syntax error: {exc.msg}") from exc

    SafeCodeValidator().visit(tree)
    return compile(tree, "<safe-python>", "exec")


def _build_env(env: dict[str, Any]) -> dict[str, Any]:
    safe_env: dict[str, Any] = {"__builtins__": dict(ALLOWED_BUILTINS)}
    # Inject pd and np so the model can use them directly without import
    safe_env["pd"] = pd
    safe_env["np"] = np
    safe_env.update(env)
    return safe_env


def _resolve_timeout_seconds(*, df: pd.DataFrame | None, mode: Literal["python"]) -> float:
    row_count = 0 if df is None else int(len(df.index))
    column_count = 0 if df is None else int(len(df.columns))
    complexity_bonus = min(4.0, row_count / 4000.0 + column_count / 25.0)
    base = SETTINGS.python_execution_timeout_base_seconds
    ceiling = SETTINGS.python_execution_timeout_max_seconds
    return min(ceiling, base + complexity_bonus)

