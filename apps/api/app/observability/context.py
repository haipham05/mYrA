"""Validated request and worker correlation context."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

_CORRELATION_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_TRACE_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")
_SPAN_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{16}$")


@dataclass(frozen=True, slots=True)
class OperationContext:
    """Opaque identifiers safe to carry between an API request and a job."""

    correlation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    sampled: bool = False

    @classmethod
    def validated(
        cls,
        *,
        correlation_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        sampled: bool = False,
    ) -> OperationContext:
        safe_correlation_id = (
            correlation_id
            if correlation_id and _CORRELATION_PATTERN.fullmatch(correlation_id)
            else None
        )
        safe_trace_id = (
            trace_id
            if trace_id
            and _TRACE_ID_PATTERN.fullmatch(trace_id)
            and trace_id.casefold() != "0" * 32
            else None
        )
        safe_span_id = (
            span_id
            if span_id and _SPAN_ID_PATTERN.fullmatch(span_id) and span_id.casefold() != "0" * 16
            else None
        )
        return cls(
            correlation_id=safe_correlation_id,
            trace_id=safe_trace_id,
            span_id=safe_span_id,
            sampled=bool(sampled and safe_trace_id and safe_span_id),
        )


_current_context: ContextVar[OperationContext] = ContextVar(
    "myra_operation_context", default=OperationContext()
)


def get_operation_context() -> OperationContext:
    return _current_context.get()


@contextmanager
def use_operation_context(context: OperationContext) -> Iterator[None]:
    """Bind context for the current async/task context and always restore it."""
    token: Token[OperationContext] = _current_context.set(context)
    try:
        yield
    finally:
        _current_context.reset(token)
