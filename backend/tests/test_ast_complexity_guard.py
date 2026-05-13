from __future__ import annotations

import ast

import pytest

from src.tools import SafeCodeValidator, SafeExecutionError


def _visit(code: str, **kwargs):
    tree = ast.parse(code, mode="exec")
    validator = SafeCodeValidator(**kwargs)
    validator.visit(tree)


def test_safe_code_under_ast_node_limit_passes():
    _visit("result = df.head()")


def test_excessive_ast_nodes_blocked():
    with pytest.raises(SafeExecutionError, match="AST 节点数超过限制"):
        _visit("a = 1\nb = 2\nc = 3\nd = 4", max_ast_nodes=5)


def test_reasonable_loop_nesting_allowed():
    _visit(
        "total = 0\nfor row in rows:\n    total += row\n",
        max_loop_nesting=1,
    )


def test_excessive_loop_nesting_blocked():
    with pytest.raises(SafeExecutionError, match="循环嵌套层数超过限制"):
        _visit(
            "for x in xs:\n    for y in ys:\n        for z in zs:\n            for w in ws:\n                pass\n",
            max_loop_nesting=3,
        )


def test_reasonable_comprehension_allowed():
    _visit("result = [x for x in values if x > 0]")


def test_excessive_comprehension_nesting_blocked():
    with pytest.raises(SafeExecutionError, match="推导式嵌套层数超过限制"):
        _visit(
            "result = [[[x for x in xs] for y in ys] for z in zs]",
            max_comprehension_nesting=2,
        )


def test_reasonable_pandas_call_chain_allowed():
    _visit('result = df.groupby("col").size().reset_index(name="count")')


def test_excessive_call_chain_blocked():
    with pytest.raises(SafeExecutionError, match="调用链过长"):
        _visit("result = a.b.c.d.e.f.g.h.i.j()")
