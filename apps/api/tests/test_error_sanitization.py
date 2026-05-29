from app.services.error_sanitizer import (
    IngestionErrorCode,
    classify_and_sanitize_error,
    redact_secrets,
)


def test_redact_secrets_cleans_credentials():
    leaked = (
        "Connection failed to postgresql://postgres:SuperSecretPassword123"
        "@db.supabase.co:5432/postgres "
        "using bearer sk-proj-1234567890abcdef1234567890 "
        "and Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    )
    clean = redact_secrets(leaked)
    assert "SuperSecretPassword123" not in clean
    assert "sk-proj-1234567890abcdef1234567890" not in clean
    assert "[REDACTED]" in clean


def test_classify_corrupt_pdf_error():
    err = ValueError("PDFSyntaxError: cannot find startxref")
    classified = classify_and_sanitize_error(err)
    assert classified.code == IngestionErrorCode.CORRUPT_OR_UNREADABLE_PDF
    assert classified.is_transient is False
    assert "corrupt" in classified.sanitized_message.lower()


def test_classify_checksum_mismatch():
    err = ValueError("Stored PDF checksum does not match the uploaded document")
    classified = classify_and_sanitize_error(err)
    assert classified.code == IngestionErrorCode.DOCUMENT_CHECKSUM_MISMATCH
    assert classified.is_transient is False


def test_classify_storage_temporary_error():
    err = TimeoutError("Connection to storage timed out")
    classified = classify_and_sanitize_error(err)
    assert classified.code == IngestionErrorCode.STORAGE_TEMPORARY_ERROR
    assert classified.is_transient is True


def test_classify_generic_internal_error_redacts_secrets():
    err = RuntimeError("Unexpected failure with sk-1234567890123456789012345 in system")
    classified = classify_and_sanitize_error(err)
    assert classified.code == IngestionErrorCode.INTERNAL_INGESTION_ERROR
    assert "sk-1234567890123456789012345" not in classified.sanitized_message
    assert "[REDACTED]" in classified.sanitized_message
