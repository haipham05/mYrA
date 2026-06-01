import enum
import re
from typing import NamedTuple


class IngestionErrorCode(enum.StrEnum):
    CORRUPT_OR_UNREADABLE_PDF = "CORRUPT_OR_UNREADABLE_PDF"
    TEXT_EXTRACTION_EMPTY = "TEXT_EXTRACTION_EMPTY"
    DOCUMENT_CHECKSUM_MISMATCH = "DOCUMENT_CHECKSUM_MISMATCH"
    STORAGE_TEMPORARY_ERROR = "STORAGE_TEMPORARY_ERROR"
    STORAGE_OBJECT_NOT_FOUND = "STORAGE_OBJECT_NOT_FOUND"
    EMBEDDING_PROVIDER_ERROR = "EMBEDDING_PROVIDER_ERROR"
    DATABASE_TRANSIENT_ERROR = "DATABASE_TRANSIENT_ERROR"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    INTERNAL_INGESTION_ERROR = "INTERNAL_INGESTION_ERROR"


class ClassifiedError(NamedTuple):
    code: IngestionErrorCode
    sanitized_message: str
    is_transient: bool


# Regex patterns to redact any leaked secrets or credentials
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"password=([^\s&]+)", re.IGNORECASE),
    re.compile(r"://([^:@\s]+):([^@\s]+)@", re.IGNORECASE),
]


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def classify_and_sanitize_error(err: Exception) -> ClassifiedError:
    err_str = str(err).lower()
    err_type = err.__class__.__name__

    if "stored pdf checksum does not match" in err_str:
        return ClassifiedError(
            code=IngestionErrorCode.DOCUMENT_CHECKSUM_MISMATCH,
            sanitized_message="Stored document checksum does not match uploaded file",
            is_transient=False,
        )

    if any(k in err_str for k in ("no extractable text", "no text found")):
        return ClassifiedError(
            code=IngestionErrorCode.TEXT_EXTRACTION_EMPTY,
            sanitized_message="Document contains no extractable text or characters",
            is_transient=False,
        )

    if any(
        k in err_str
        for k in (
            "corrupt",
            "not a pdf",
            "invalid pdf",
            "unsupported format",
            "pdfsyntaxerror",
            "formaterror",
            "cannot find startxref",
        )
    ):
        return ClassifiedError(
            code=IngestionErrorCode.CORRUPT_OR_UNREADABLE_PDF,
            sanitized_message="PDF file format is corrupt, malformed, or unreadable",
            is_transient=False,
        )

    if isinstance(err, FileNotFoundError) or "not found" in err_str:
        return ClassifiedError(
            code=IngestionErrorCode.STORAGE_OBJECT_NOT_FOUND,
            sanitized_message="Document object was not found in storage backend",
            is_transient=False,
        )

    if any(k in err_str for k in ("ocr", "parse", "parsing", "chunk")) and (
        isinstance(err, TimeoutError) or "timeout" in err_str
    ):
        return ClassifiedError(
            code=IngestionErrorCode.PROCESSING_TIMEOUT,
            sanitized_message="Document parsing or OCR processing timed out",
            is_transient=True,
        )

    if any(k in err_str for k in ("embedding", "sentence-transformers", "model")):
        return ClassifiedError(
            code=IngestionErrorCode.EMBEDDING_PROVIDER_ERROR,
            sanitized_message="Document embedding generation service error",
            is_transient=True,
        )

    if isinstance(err, (TimeoutError, ConnectionError, OSError)) or any(
        k in err_type for k in ("Connection", "Timeout", "Network")
    ):
        return ClassifiedError(
            code=IngestionErrorCode.STORAGE_TEMPORARY_ERROR,
            sanitized_message="Temporary network or storage connectivity error",
            is_transient=True,
        )

    if any(k in err_type for k in ("OperationalError", "DatabaseError", "InternalError")):
        return ClassifiedError(
            code=IngestionErrorCode.DATABASE_TRANSIENT_ERROR,
            sanitized_message="Temporary database operational error",
            is_transient=True,
        )

    # Fallback generic error
    safe_msg = redact_secrets(str(err))
    clean_msg = " ".join(safe_msg.splitlines())[:200]
    return ClassifiedError(
        code=IngestionErrorCode.INTERNAL_INGESTION_ERROR,
        sanitized_message=f"Ingestion processing error: {clean_msg}",
        is_transient=False,
    )
