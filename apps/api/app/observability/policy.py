"""Conservative allowlist-compatible sanitization for observation payloads."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

MAX_OBSERVATION_BYTES = 128 * 1024
_MAX_STRING_CHARS = MAX_OBSERVATION_BYTES
_MAX_COLLECTION_ITEMS = 100
_MAX_NESTING = 8
_SENSITIVE_KEY = re.compile(
    r"(?:secret|password|authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|cookie)",
    re.IGNORECASE,
)
_BEARER_VALUE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_AUTHORIZATION_VALUE = re.compile(r"(?i)\b(authorization\s*[=:]\s*)[^;\r\n]*")
_KEY_VALUE = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)"
    r"(\s*[=:]\s*)([^\s&,;]+)"
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d().\s-]{7,}\d)(?!\w)")
_URL = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_SENSITIVE_QUERY_KEY = re.compile(
    r"(?:token|key|secret|signature|credential|password|auth)", re.IGNORECASE
)


def sanitize_text(value: str) -> str:
    """Remove common credentials/contact identifiers and bound any one string.

    Sanitization is a precaution, not anonymization. Callers must only provide
    deliberately selected application content, never arbitrary objects or headers.
    """

    def sanitize_url(match: re.Match[str]) -> str:
        raw_url = match.group(0)
        try:
            parsed = urlsplit(raw_url)
            hostname = parsed.hostname or ""
            if ":" in hostname and not hostname.startswith("["):
                hostname = f"[{hostname}]"
            netloc = hostname
            if parsed.port:
                netloc = f"{netloc}:{parsed.port}"
            query = urlencode(
                [
                    (key, "[REDACTED]" if _SENSITIVE_QUERY_KEY.search(key) else item)
                    for key, item in parse_qsl(parsed.query, keep_blank_values=True)
                ]
            )
            return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
        except ValueError:
            return "[URL REDACTED]"

    # Sanitize URLs before contact patterns: URL userinfo can resemble an email.
    redacted = _URL.sub(sanitize_url, value)
    redacted = _AUTHORIZATION_VALUE.sub(r"\1[REDACTED]", redacted)
    redacted = _BEARER_VALUE.sub("Bearer [REDACTED]", redacted)
    redacted = _KEY_VALUE.sub(r"\1\2[REDACTED]", redacted)
    redacted = _EMAIL.sub("[EMAIL REDACTED]", redacted)
    redacted = _PHONE.sub("[PHONE REDACTED]", redacted)
    if len(redacted) > _MAX_STRING_CHARS:
        return redacted[:_MAX_STRING_CHARS] + "[TRUNCATED]"
    return redacted


def _sanitize_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= _MAX_NESTING:
        return "[MAX DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_COLLECTION_ITEMS:
                safe["_truncated"] = True
                break
            safe_key = sanitize_text(str(key))[:128]
            safe[safe_key] = (
                "[REDACTED]"
                if _SENSITIVE_KEY.search(safe_key)
                else _sanitize_value(item, depth=depth + 1)
            )
        return safe
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = list(value[:_MAX_COLLECTION_ITEMS])
        safe_items = [_sanitize_value(item, depth=depth + 1) for item in items]
        if len(value) > _MAX_COLLECTION_ITEMS:
            safe_items.append("[TRUNCATED]")
        return safe_items
    return "[UNSUPPORTED VALUE]"


def sanitize_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitize a deliberately selected payload and cap its serialized size."""
    safe = _sanitize_value(value)
    if not isinstance(safe, dict):
        return {"payload": "[UNSUPPORTED VALUE]"}
    encoded = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) <= MAX_OBSERVATION_BYTES:
        return safe

    # Keep correlation/stage fields first; text-bearing values are safely elided
    # rather than slicing JSON or emitting an invalid payload.
    compact: dict[str, Any] = {}
    for key, item in safe.items():
        candidate = dict(compact)
        candidate[key] = item
        if key in {"input", "output", "evidence", "prompt", "answer"}:
            candidate[key] = "[TRUNCATED]"
        candidate["_truncated"] = True
        if (
            len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            <= MAX_OBSERVATION_BYTES
        ):
            compact = candidate
        else:
            compact["_truncated"] = True
    final = json.dumps(compact, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(final) > MAX_OBSERVATION_BYTES:
        return {"_truncated": True, "payload": "[TRUNCATED]"}
    return compact
