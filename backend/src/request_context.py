from __future__ import annotations

import contextlib
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterator
from uuid import uuid4


@dataclass(slots=True)
class RequestContextState:
    request_id: str
    degradation_mode: str = "none"
    failure_stage: str | None = None
    route_diagnostics: dict[str, Any] | None = None


CURRENT_REQUEST_CONTEXT: ContextVar[RequestContextState | None] = ContextVar(
    "current_request_context",
    default=None,
)


def create_request_id() -> str:
    return str(uuid4())


def get_request_context() -> RequestContextState | None:
    return CURRENT_REQUEST_CONTEXT.get()


def get_request_id() -> str | None:
    context = get_request_context()
    return context.request_id if context else None


def get_degradation_mode() -> str:
    context = get_request_context()
    return context.degradation_mode if context else "none"


def get_failure_stage() -> str | None:
    context = get_request_context()
    return context.failure_stage if context else None


def get_route_diagnostics() -> dict[str, Any] | None:
    context = get_request_context()
    if context is None or context.route_diagnostics is None:
        return None
    return dict(context.route_diagnostics)


def set_degradation_mode(mode: str) -> None:
    context = get_request_context()
    if context is not None:
        context.degradation_mode = mode


def set_failure_stage(stage: str | None) -> None:
    context = get_request_context()
    if context is not None:
        context.failure_stage = stage


def set_route_diagnostics(payload: dict[str, Any] | None) -> None:
    context = get_request_context()
    if context is not None:
        context.route_diagnostics = dict(payload) if isinstance(payload, dict) else None


@contextlib.contextmanager
def bind_request_context(
    request_id: str,
    *,
    degradation_mode: str = "none",
    failure_stage: str | None = None,
    route_diagnostics: dict[str, Any] | None = None,
) -> Iterator[RequestContextState]:
    state = RequestContextState(
        request_id=request_id,
        degradation_mode=degradation_mode,
        failure_stage=failure_stage,
        route_diagnostics=dict(route_diagnostics) if isinstance(route_diagnostics, dict) else None,
    )
    token: Token[RequestContextState | None] = CURRENT_REQUEST_CONTEXT.set(state)
    try:
        yield state
    finally:
        CURRENT_REQUEST_CONTEXT.reset(token)
