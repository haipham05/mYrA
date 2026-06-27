from concurrent.futures import ThreadPoolExecutor

from app.observability import (
    OperationContext,
    get_operation_context,
    sanitize_observation,
    sanitize_text,
    use_operation_context,
)
from app.observability.policy import MAX_OBSERVATION_BYTES


def test_operation_context_validates_identifiers_and_sample_flag() -> None:
    valid = OperationContext.validated(
        correlation_id="request-123",
        trace_id="a" * 32,
        span_id="b" * 16,
        sampled=True,
    )
    assert valid.sampled is True
    assert valid.trace_id == "a" * 32

    invalid = OperationContext.validated(
        correlation_id="Bearer secret",
        trace_id="0" * 32,
        span_id="bad",
        sampled=True,
    )
    assert invalid.correlation_id is None
    assert invalid.trace_id is None
    assert invalid.span_id is None
    assert invalid.sampled is False


def test_operation_context_resets_after_nested_and_failed_scope() -> None:
    outer = OperationContext.validated(correlation_id="outer")
    inner = OperationContext.validated(correlation_id="inner")

    with use_operation_context(outer):
        assert get_operation_context() == outer
        try:
            with use_operation_context(inner):
                assert get_operation_context() == inner
                raise RuntimeError("expected")
        except RuntimeError:
            pass
        assert get_operation_context() == outer

    assert get_operation_context() == OperationContext()


def test_context_does_not_leak_between_threads() -> None:
    def read_context(correlation_id: str) -> str | None:
        context = OperationContext.validated(correlation_id=correlation_id)
        with use_operation_context(context):
            return get_operation_context().correlation_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(read_context, [f"request-{i}" for i in range(8)]))
    assert results == [f"request-{i}" for i in range(8)]


def test_sanitizer_redacts_credentials_contacts_and_url_userinfo() -> None:
    text = (
        "Bearer abc.def; api_key=secret-value; owner@example.com; "
        "+1 (415) 555-0182; https://user:pass@example.test/a?token=secret&x=ok"
    )
    result = sanitize_text(text)
    assert "abc.def" not in result
    assert "secret-value" not in result
    assert "owner@example.com" not in result
    assert "415) 555-0182" not in result
    assert "user:pass" not in result
    assert "token=%5BREDACTED%5D" in result
    assert "x=ok" in result


def test_sanitizer_redacts_basic_authorization_headers_and_key_values() -> None:
    result = sanitize_text("Authorization: Basic dXNlcjpwYXNzd29yZA==; authorization=secret-token")
    assert "dXNlcjpwYXNzd29yZA==" not in result
    assert "secret-token" not in result
    assert result.count("[REDACTED]") == 2


def test_sanitizer_redacts_complete_digest_authorization_value() -> None:
    result = sanitize_text(
        "Authorization: Digest username=alice, response=secret, signature=abc; request accepted"
    )
    assert "username=alice" not in result
    assert "response=secret" not in result
    assert "signature=abc" not in result
    assert "request accepted" in result


def test_observation_sanitizes_sensitive_keys_and_bounds_serialized_size() -> None:
    result = sanitize_observation(
        {
            "trace_id": "a" * 32,
            "api_key": "do-not-keep",
            "output": "safe answer " + "x" * (MAX_OBSERVATION_BYTES * 2),
        }
    )
    assert result["api_key"] == "[REDACTED]"
    assert result["_truncated"] is True
    assert result["output"] == "[TRUNCATED]"
