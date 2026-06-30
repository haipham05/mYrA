from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import pytest

from app.observability.context import (
    OperationContext,
    get_operation_context,
    use_operation_context,
)
from app.observability.policy import MAX_OBSERVATION_BYTES
from app.observability.telemetry import (
    TelemetryAdapter,
    TelemetryConfig,
    _FailureCountingExporter,
)


class FakeObservation:
    def __init__(self, *, recording: bool = True) -> None:
        self.updates: list[dict[str, Any]] = []
        self.trace_id = "c" * 32
        self.id = "d" * 16
        self._otel_span = FakeOtelSpan(recording)

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeOtelSpan:
    def __init__(self, recording: bool) -> None:
        self.recording = recording

    def is_recording(self) -> bool:
        return self.recording


class FakeClient:
    def __init__(
        self,
        *,
        fail_start: bool = False,
        fail_exit: bool = False,
        recording: bool = True,
    ) -> None:
        self.fail_start = fail_start
        self.fail_exit = fail_exit
        self.calls: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.flush_count = 0
        self.shutdown_count = 0
        self.observation = FakeObservation(recording=recording)

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any):
        if self.fail_start:
            raise OSError("export target includes secret=do-not-log")
        self.calls.append(kwargs)
        try:
            yield self.observation
        finally:
            if self.fail_exit:
                raise OSError("export target includes secret=do-not-log")

    def create_event(self, **kwargs: Any) -> None:
        self.events.append(kwargs)

    def flush(self) -> None:
        self.flush_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1


def configured(**overrides: Any) -> TelemetryConfig:
    fields: dict[str, Any] = {
        "enabled": True,
        "base_url": "http://langfuse:3000",
        "public_key": "pk-test",
        "secret_key": "sk-test",
    }
    fields.update(overrides)
    return TelemetryConfig(**fields)


def test_disabled_adapter_does_not_initialize_client_or_make_network_attempt() -> None:
    attempts = 0

    def factory(_: TelemetryConfig) -> FakeClient:
        nonlocal attempts
        attempts += 1
        raise AssertionError("disabled mode must not initialize an exporter")

    adapter = TelemetryAdapter(config=configured(enabled=False), client_factory=factory)
    with adapter.operation("chat") as observation:
        assert observation is None
    adapter.event("chat.completed")
    adapter.flush()
    assert attempts == 0
    assert not adapter.enabled


def test_incomplete_configuration_stays_noop() -> None:
    attempts = 0

    def factory(_: TelemetryConfig) -> FakeClient:
        nonlocal attempts
        attempts += 1
        return FakeClient()

    adapter = TelemetryAdapter(config=configured(secret_key=None), client_factory=factory)
    assert attempts == 0
    assert adapter.dropped_count == 0


def test_stage_sanitizes_explicit_payload_and_propagates_valid_context() -> None:
    client = FakeClient()
    adapter = TelemetryAdapter(client, config=configured())
    context = OperationContext.validated(
        correlation_id="req-42",
        trace_id="a" * 32,
        span_id="b" * 16,
        sampled=True,
    )

    with use_operation_context(context):
        with adapter.stage(
            "chat.generate",
            input={
                "question": "What is attention?",
                "authorization": "Bearer do-not-export",
                "callback": "person@example.com",
            },
            metadata={"stage": "generation"},
            generation=True,
        ) as span:
            adapter.generation_metadata(
                span,
                model="deepseek-chat",
                usage={"input": 12, "output": 5},
                output="An answer for person@example.com",
            )

    call = client.calls[0]
    assert call["as_type"] == "generation"
    assert call["trace_context"] == {
        "trace_id": "a" * 32,
        "parent_span_id": "b" * 16,
    }
    assert call["metadata"]["correlation_id"] == "req-42"
    assert call["input"]["question"] == "What is attention?"
    assert call["input"]["authorization"] == "[REDACTED]"
    assert call["input"]["callback"] == "[EMAIL REDACTED]"
    assert client.observation.updates[0]["output"] == "An answer for [EMAIL REDACTED]"
    assert client.observation.updates[0]["usage_details"] == {"input": 12, "output": 5}


def test_payload_is_bounded_to_128_kibibytes() -> None:
    client = FakeClient()
    adapter = TelemetryAdapter(client, config=configured())
    with adapter.stage("large.input", input={"question": "q" * 200_000}):
        pass
    encoded = json.dumps(client.calls[0]["input"], separators=(",", ":")).encode()
    assert len(encoded) <= MAX_OBSERVATION_BYTES
    assert "TRUNCATED" in str(client.calls[0]["input"])


def test_start_and_update_share_one_128_kibibyte_observation_budget() -> None:
    client = FakeClient()
    adapter = TelemetryAdapter(client, config=configured())
    with adapter.stage(
        "large.lifecycle",
        input={"question": "q" * 50_000},
        metadata={"selected": "m" * 10_000},
    ) as observation:
        assert observation is not None
        observation.update(output="a" * 80_000, metadata={"detail": "d" * 40_000})

    call = client.calls[0]
    observed_payload = {
        "initial": {key: call[key] for key in ("input", "metadata") if key in call},
        "updates": client.observation.updates,
    }
    serialized = json.dumps(observed_payload, ensure_ascii=False, separators=(",", ":")).encode()
    assert len(serialized) <= MAX_OBSERVATION_BYTES
    assert "_truncated" in str(observed_payload)


def test_product_exception_remains_identical_and_context_is_restored() -> None:
    client = FakeClient()
    adapter = TelemetryAdapter(client, config=configured())
    initial = get_operation_context()
    expected = RuntimeError("product failure")
    context = OperationContext.validated(correlation_id="operation-1")

    with use_operation_context(context):
        with pytest.raises(RuntimeError) as caught:
            with adapter.operation("failing-operation"):
                raise expected
        assert caught.value is expected
        assert get_operation_context() == context

    assert get_operation_context() == initial
    assert client.observation.updates[0]["level"] == "ERROR"
    assert client.observation.updates[0]["status_message"] == "RuntimeError"

    exporter_failure = FakeClient(fail_exit=True)
    adapter = TelemetryAdapter(exporter_failure, config=configured())
    product_error = ValueError("original product error")
    with pytest.raises(ValueError) as caught:
        with adapter.operation("body-and-export-failure"):
            raise product_error
    assert caught.value is product_error


def test_sdk_observation_context_populates_and_restores_durable_trace_context() -> None:
    client = FakeClient()
    adapter = TelemetryAdapter(client, config=configured())
    initial = OperationContext.validated(correlation_id="request-4")

    with use_operation_context(initial):
        with adapter.operation("request"):
            active = get_operation_context()
            assert active.correlation_id == "request-4"
            assert active.trace_id == "c" * 32
            assert active.span_id == "d" * 16
            assert active.sampled
        assert get_operation_context() == initial


def test_non_recording_root_remains_unsampled_through_nested_stage() -> None:
    client = FakeClient(recording=False)
    adapter = TelemetryAdapter(client, config=configured())

    with adapter.operation("unsampled-root"):
        root_context = get_operation_context()
        assert root_context.trace_id == "c" * 32
        assert root_context.span_id == "d" * 16
        assert not root_context.sampled
        with adapter.stage("nested-stage") as nested:
            assert nested is None

    assert len(client.calls) == 1


def test_exporter_start_and_exit_failures_are_dropped_without_affecting_work() -> None:
    failed_start = FakeClient(fail_start=True)
    adapter = TelemetryAdapter(failed_start, config=configured())
    with adapter.operation("start-failure") as observation:
        assert observation is None
        result = "product-result"
    assert result == "product-result"
    assert adapter.dropped_count == 1

    failed_exit = FakeClient(fail_exit=True)
    adapter = TelemetryAdapter(failed_exit, config=configured())
    with adapter.operation("exit-failure") as observation:
        assert observation is not None
        result = "still-successful"
    assert result == "still-successful"
    assert adapter.dropped_count == 1


def test_events_and_lifecycle_failures_are_isolated_and_context_does_not_leak() -> None:
    client = FakeClient()
    adapter = TelemetryAdapter(client, config=configured())
    with use_operation_context(OperationContext.validated(correlation_id="one")):
        adapter.event("job.finished", metadata={"attempt": 1})
    with use_operation_context(OperationContext.validated(correlation_id="two")):
        adapter.event("job.finished", metadata={"attempt": 2})
    assert client.events[0]["metadata"]["correlation_id"] == "one"
    assert client.events[1]["metadata"]["correlation_id"] == "two"

    client.flush = lambda: (_ for _ in ()).throw(OSError("secret=hidden"))  # type: ignore[method-assign]
    client.shutdown = lambda: (_ for _ in ()).throw(OSError("secret=hidden"))  # type: ignore[method-assign]
    adapter.flush()
    adapter.shutdown()
    assert adapter.dropped_count == 2


def test_async_export_failures_are_counted_without_raising() -> None:
    failures = 0

    def on_failure() -> None:
        nonlocal failures
        failures += 1

    class BrokenExporter:
        def export(self, spans: Any) -> None:
            raise OSError("secret=not-included")

        def force_flush(self, timeout_millis: int) -> bool:
            return False

        def shutdown(self) -> None:
            raise OSError("secret=not-included")

    exporter = _FailureCountingExporter(BrokenExporter(), on_failure=on_failure)
    assert exporter.export([]) is None
    assert not exporter.force_flush()
    exporter.shutdown()
    assert failures == 3
