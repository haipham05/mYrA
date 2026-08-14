"""Explicit, failure-isolated Langfuse adapter for selected application data.

Nothing is exported unless the adapter is explicitly enabled and has complete
credentials. Callers must pass only deliberately selected fields; this module
never inspects function arguments, ORM objects, request headers, or exceptions.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from app.observability.context import OperationContext, get_operation_context, use_operation_context
from app.observability.policy import MAX_OBSERVATION_BYTES, sanitize_observation, sanitize_text

logger = logging.getLogger(__name__)


class Observation(Protocol):
    def update(self, **kwargs: Any) -> Any: ...


class LangfuseClient(Protocol):
    def start_as_current_observation(self, **kwargs: Any) -> Any: ...

    def create_event(self, **kwargs: Any) -> Any: ...

    def flush(self) -> Any: ...

    def shutdown(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    enabled: bool
    base_url: str | None = None
    public_key: str | None = None
    secret_key: str | None = None
    sample_rate: float = 1.0
    timeout_seconds: int = 2
    flush_at: int = 64
    flush_interval_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> TelemetryConfig:
        enabled = os.getenv("MYRA_OBSERVABILITY_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        sample_rate = _bounded_float(os.getenv("MYRA_TRACE_SAMPLE_RATE", "1"), 1.0)
        return cls(
            enabled=enabled,
            base_url=os.getenv("LANGFUSE_BASE_URL"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            sample_rate=min(1.0, max(0.0, sample_rate)),
        )

    @property
    def ready(self) -> bool:
        return bool(self.enabled and self.base_url and self.public_key and self.secret_key)


def _bounded_float(value: str, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed == parsed and abs(parsed) != float("inf") else fallback


def _no_op_mask(*, params: Any) -> Any:
    """Export-boundary defense for any accidentally emitted string attributes."""
    try:
        from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

        patches = {}
        for identifier, span in params.spans.items():
            replacements = {
                key: sanitize_text(value)
                for key, value in span.attributes.items()
                if isinstance(value, str) and sanitize_text(value) != value
            }
            if replacements:
                patches[identifier] = OtelSpanPatch(set_attributes=replacements)
        return MaskOtelSpansResult(span_patches=patches) if patches else None
    except Exception:
        # Masking is a final defense, never a reason to change a product result.
        return None


def _otlp_endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Langfuse base URL must be an HTTP(S) URL")
    base_path = parsed.path.rstrip("/")
    path = f"{base_path}/api/public/otel/v1/traces"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _observation_is_recording(observation: Observation) -> bool:
    """Use actual OTel sampling state; IDs alone survive non-recording spans."""
    otel_span = getattr(observation, "_otel_span", None)
    candidates = (otel_span, observation) if otel_span is not None else (observation,)
    for candidate in candidates:
        is_recording = getattr(candidate, "is_recording", None)
        if callable(is_recording):
            try:
                if not bool(is_recording()):
                    return False
                return True
            except Exception:
                return False

        get_span_context = getattr(candidate, "get_span_context", None)
        if callable(get_span_context):
            try:
                span_context = get_span_context()
                trace_flags = getattr(span_context, "trace_flags", None)
                sampled_flag = getattr(trace_flags, "sampled", None)
                if sampled_flag is not None:
                    return bool(sampled_flag)
            except Exception:
                return False

    # Fail closed if this SDK shape exposes no trustworthy sampling state.
    return False


def _encoded_size(value: Mapping[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


class _ObservationPayloadBudget:
    """Count initial and incremental content against one observation limit."""

    def __init__(self, initial: Mapping[str, Any]) -> None:
        self.used_bytes = _encoded_size(initial)

    def fit(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        safe = sanitize_observation(payload)
        available = max(0, MAX_OBSERVATION_BYTES - self.used_bytes)
        if _encoded_size(safe) > available:
            compact = dict(safe)
            for key in ("input", "output", "evidence", "prompt", "answer"):
                if key in compact:
                    compact[key] = "[TRUNCATED]"
            compact["_truncated"] = True
            safe = compact
        if _encoded_size(safe) > available:
            safe = {"_truncated": True, "payload": "[TRUNCATED]"}
        if _encoded_size(safe) <= available:
            self.used_bytes += _encoded_size(safe)
            return safe
        # If even the marker cannot fit, omit this update rather than exceed cap.
        return {}


class _BoundedObservation:
    """Langfuse observation proxy enforcing a cumulative serialized-size cap."""

    def __init__(
        self,
        observation: Observation,
        budget: _ObservationPayloadBudget,
        *,
        on_update_failure: Callable[[BaseException], None],
    ) -> None:
        self._observation = observation
        self._budget = budget
        self._on_update_failure = on_update_failure

    @property
    def trace_id(self) -> str | None:
        return getattr(self._observation, "trace_id", None)

    @property
    def id(self) -> str | None:
        return getattr(self._observation, "id", None)

    @property
    def _otel_span(self) -> Any:
        return getattr(self._observation, "_otel_span", None)

    def update(self, **kwargs: Any) -> Any:
        safe = self._budget.fit(kwargs)
        if safe:
            try:
                return self._observation.update(**safe)
            except Exception as exc:
                self._on_update_failure(exc)
        return None


def _build_client(
    config: TelemetryConfig,
    *,
    on_export_failure: Callable[[], None] | None = None,
) -> LangfuseClient:
    """Build an isolated, bounded client only after explicit opt-in."""
    from langfuse import Langfuse
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider

    assert config.base_url and config.public_key and config.secret_key
    auth = base64.b64encode(f"{config.public_key}:{config.secret_key}".encode()).decode()
    exporter = _FailureCountingExporter(
        OTLPSpanExporter(
            endpoint=_otlp_endpoint(config.base_url),
            headers={
                "Authorization": f"Basic {auth}",
                "x-langfuse-ingestion-version": "4",
            },
            timeout=config.timeout_seconds,
        ),
        on_failure=on_export_failure,
    )
    return Langfuse(
        public_key=config.public_key,
        secret_key=config.secret_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
        mask_otel_spans=_no_op_mask,
        sample_rate=config.sample_rate,
        flush_at=config.flush_at,
        flush_interval=config.flush_interval_seconds,
    )


class _AtomicDropCount:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def increment(self) -> None:
        with self._lock:
            self._value += 1


class _FailureCountingExporter:
    """Count asynchronous OTLP failures while keeping exceptions off app paths."""

    def __init__(self, exporter: Any, *, on_failure: Callable[[], None] | None) -> None:
        self._exporter = exporter
        self._on_failure = on_failure

    def _failed(self) -> None:
        if self._on_failure is not None:
            self._on_failure()

    def export(self, spans: Any) -> Any:
        try:
            result = self._exporter.export(spans)
        except Exception:
            self._failed()
            return None
        if getattr(result, "name", "") != "SUCCESS":
            self._failed()
        return result

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        try:
            result = self._exporter.force_flush(timeout_millis)
        except Exception:
            self._failed()
            return False
        if result is False:
            self._failed()
        return result

    def shutdown(self) -> None:
        try:
            self._exporter.shutdown()
        except Exception:
            self._failed()


class TelemetryAdapter:
    """No-op-by-default observation API that never raises into product code."""

    def __init__(
        self,
        client: LangfuseClient | None = None,
        *,
        config: TelemetryConfig | None = None,
        client_factory: Callable[[TelemetryConfig], LangfuseClient] = _build_client,
    ) -> None:
        self._config = config or TelemetryConfig.from_env()
        self._client = client
        self._dropped = 0
        self._export_drops = _AtomicDropCount()
        if self._client is None and self._config.ready:
            try:
                if client_factory is _build_client:
                    self._client = _build_client(
                        self._config, on_export_failure=self._record_export_failure
                    )
                else:
                    self._client = client_factory(self._config)
            except Exception as exc:
                self._drop("client_initialization", exc)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def dropped_count(self) -> int:
        return self._dropped + self._export_drops.value

    def _record_export_failure(self) -> None:
        self._export_drops.increment()
        logger.warning(
            "Telemetry export batch was dropped",
            extra={"action": "export_batch", "outcome": "dropped"},
        )

    def _drop(self, action: str, error: BaseException) -> None:
        self._dropped += 1
        # Only the exception class is safe diagnostic metadata; omit error text.
        logger.warning(
            "Telemetry operation was dropped",
            extra={"action": action, "outcome": "dropped", "error_code": type(error).__name__},
        )

    @staticmethod
    def _safe_payload(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
        return sanitize_observation(value) if value is not None else None

    @staticmethod
    def _safe_name(name: str) -> str:
        # Stable operation names are metadata, never arbitrary user text.
        safe = sanitize_text(name.strip())[:100]
        if not safe or any(character in safe for character in "\r\n"):
            return "unnamed-operation"
        return safe

    @contextmanager
    def operation(
        self,
        name: str,
        *,
        input: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[Observation | None]:
        with self._observation("span", name, input=input, metadata=metadata) as span:
            yield span

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        input: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        generation: bool = False,
    ) -> Iterator[Observation | None]:
        with self._observation(
            "generation" if generation else "span", name, input=input, metadata=metadata
        ) as span:
            yield span

    @contextmanager
    def _observation(
        self,
        as_type: str,
        name: str,
        *,
        input: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None,
    ) -> Iterator[Observation | None]:
        client = self._client
        if client is None:
            yield None
            return

        context = get_operation_context()
        raw_fields: dict[str, Any] = {}
        if metadata is not None:
            raw_fields["metadata"] = dict(metadata)
        elif context.correlation_id:
            raw_fields["metadata"] = {}
        if context.correlation_id:
            raw_fields["metadata"]["correlation_id"] = context.correlation_id
        if input is not None:
            raw_fields["input"] = input
        safe_fields = sanitize_observation(raw_fields)
        safe_metadata = safe_fields.get("metadata", {})
        if not isinstance(safe_metadata, dict):
            safe_metadata = {"truncated": True}
        if safe_fields.get("_truncated"):
            safe_metadata["_truncated"] = True
        trace_context: dict[str, str] = {}
        if context.trace_id and context.sampled:
            trace_context["trace_id"] = context.trace_id
            if context.span_id:
                trace_context["parent_span_id"] = context.span_id
        elif context.trace_id and not context.sampled:
            # A persisted unsampled job must stay unsampled on every retry.
            yield None
            return

        kwargs: dict[str, Any] = {
            "as_type": as_type,
            "name": self._safe_name(name),
            "metadata": safe_metadata or None,
        }
        safe_input = safe_fields.get("input")
        if safe_input is not None:
            kwargs["input"] = safe_input
        if trace_context:
            kwargs["trace_context"] = trace_context

        try:
            observation_context = client.start_as_current_observation(**kwargs)
        except Exception as exc:
            self._drop("observation_start", exc)
            yield None
            return

        product_exception: BaseException | None = None
        observation_yielded = False
        try:
            with observation_context as observation:
                observed_trace_id = getattr(observation, "trace_id", None)
                observed_span_id = getattr(observation, "id", None)
                active_context = OperationContext.validated(
                    correlation_id=context.correlation_id,
                    trace_id=observed_trace_id,
                    span_id=observed_span_id,
                    sampled=bool(
                        observed_trace_id
                        and observed_span_id
                        and _observation_is_recording(observation)
                    ),
                )
                initial_payload: dict[str, Any] = {"metadata": safe_metadata}
                if safe_input is not None:
                    initial_payload["input"] = safe_input
                bounded_observation = _BoundedObservation(
                    observation,
                    _ObservationPayloadBudget(initial_payload),
                    on_update_failure=lambda exc: self._drop("observation_update", exc),
                )
                with use_operation_context(active_context):
                    try:
                        observation_yielded = True
                        yield bounded_observation
                    except BaseException as exc:
                        product_exception = exc
                        self._update_safely(
                            observation, level="ERROR", status_message=type(exc).__name__
                        )
                        raise
        except Exception as exc:
            if product_exception is not None:
                if exc is product_exception:
                    raise
                # Preserve the exact product/cancellation exception if SDK
                # teardown raises while that exception is already unwinding.
                raise product_exception.with_traceback(product_exception.__traceback__)
            # An SDK context-manager/export failure must not alter product work.
            self._drop("observation", exc)
            if not observation_yielded:
                yield None

    def _update_safely(self, observation: Observation | None, **kwargs: Any) -> None:
        if observation is None:
            return
        try:
            safe_kwargs = dict(kwargs)
            bounded_fields = {
                key: safe_kwargs.pop(key)
                for key in ("input", "output", "metadata")
                if key in safe_kwargs and safe_kwargs[key] is not None
            }
            safe_kwargs.update(sanitize_observation(bounded_fields))
            observation.update(**safe_kwargs)
        except Exception as exc:
            self._drop("observation_update", exc)

    def event(
        self,
        name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        input: Mapping[str, Any] | None = None,
        output: Mapping[str, Any] | None = None,
    ) -> None:
        client = self._client
        if client is None:
            return
        context = get_operation_context()
        if context.trace_id and not context.sampled:
            # Do not turn an explicitly unsampled persisted job into a new trace.
            return
        trace_context: dict[str, str] = {}
        if context.trace_id and context.sampled:
            trace_context = {"trace_id": context.trace_id}
            if context.span_id:
                trace_context["parent_span_id"] = context.span_id
        raw_fields: dict[str, Any] = {}
        if metadata is not None:
            raw_fields["metadata"] = dict(metadata)
        elif context.correlation_id:
            raw_fields["metadata"] = {}
        if context.correlation_id:
            raw_fields["metadata"]["correlation_id"] = context.correlation_id
        if input is not None:
            raw_fields["input"] = input
        if output is not None:
            raw_fields["output"] = output
        safe_fields = sanitize_observation(raw_fields)
        safe_metadata = safe_fields.get("metadata", {})
        if not isinstance(safe_metadata, dict):
            safe_metadata = {"truncated": True}
        try:
            client.create_event(
                name=self._safe_name(name),
                metadata=safe_metadata or None,
                input=safe_fields.get("input"),
                output=safe_fields.get("output"),
                trace_context=trace_context or None,
            )
        except Exception as exc:
            self._drop("event", exc)

    def generation_metadata(
        self,
        observation: Observation | None,
        *,
        model: str | None,
        usage: Mapping[str, int] | None,
        output: str | None = None,
    ) -> None:
        """Attach explicitly supplied provider metadata; never infer/calculate cost."""
        metadata: dict[str, Any] = {}
        if usage is not None:
            metadata["usage"] = dict(usage)
        kwargs: dict[str, Any] = {"metadata": metadata or None}
        if model:
            kwargs["model"] = sanitize_text(model)
        if usage:
            kwargs["usage_details"] = {
                key: value for key, value in usage.items() if isinstance(value, int)
            }
        if output is not None:
            kwargs["output"] = sanitize_text(output)
        self._update_safely(observation, **kwargs)

    def flush(self) -> None:
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:
            self._drop("flush", exc)

    def shutdown(self) -> None:
        if self._client is None:
            return
        try:
            self._client.shutdown()
        except Exception as exc:
            self._drop("shutdown", exc)


_default_telemetry: TelemetryAdapter | None = None
_default_telemetry_lock = threading.Lock()


def get_telemetry() -> TelemetryAdapter:
    """Return the process-wide lazy adapter, safe for concurrent first access."""
    global _default_telemetry
    if _default_telemetry is None:
        with _default_telemetry_lock:
            if _default_telemetry is None:
                _default_telemetry = TelemetryAdapter()
    return _default_telemetry
