"""Failure-isolated JSON cache used only for disposable derived results."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from app.config import Settings

logger = logging.getLogger(__name__)
T = TypeVar("T")
MAX_CACHE_ENTRY_BYTES = 1024 * 1024


class RedisClient(Protocol):
    def get(self, name: str) -> str | bytes | None: ...

    def set(self, name: str, value: str, *, ex: int) -> Any: ...


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    errors: int = 0


class JsonCache:
    """A bounded JSON cache. Callers provide a validator for the expected value type."""

    def __init__(
        self,
        client: RedisClient | None = None,
        *,
        enabled: bool = True,
        max_entry_bytes: int = MAX_CACHE_ENTRY_BYTES,
    ) -> None:
        self._client = client
        self._enabled = enabled and client is not None
        self._max_entry_bytes = max_entry_bytes
        self.stats = CacheStats()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def _redis_key(key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"myra:v1:{digest}"

    def get(self, key: str, validator: Callable[[object], T]) -> T | None:
        if not self._enabled or self._client is None:
            self.stats.misses += 1
            return None
        try:
            encoded = self._client.get(self._redis_key(key))
            if encoded is None:
                self.stats.misses += 1
                return None
            if isinstance(encoded, bytes):
                if len(encoded) > self._max_entry_bytes:
                    self.stats.misses += 1
                    return None
                encoded = encoded.decode("utf-8")
            elif len(encoded.encode("utf-8")) > self._max_entry_bytes:
                self.stats.misses += 1
                return None
            envelope = json.loads(encoded)
            if not isinstance(envelope, Mapping) or envelope.get("version") != 1:
                self.stats.misses += 1
                return None
            value = validator(envelope.get("value"))
        except Exception as exc:
            self.stats.errors += 1
            logger.info(
                "cache read unavailable; recomputing", extra={"error_type": type(exc).__name__}
            )
            return None
        self.stats.hits += 1
        return value

    def set(self, key: str, value: object, *, ttl_seconds: int) -> bool:
        if not self._enabled or self._client is None or ttl_seconds <= 0:
            return False
        try:
            encoded = json.dumps(
                {"version": 1, "value": value},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            if len(encoded.encode("utf-8")) > self._max_entry_bytes:
                self.stats.misses += 1
                return False
            self._client.set(self._redis_key(key), encoded, ex=ttl_seconds)
            return True
        except Exception as exc:
            self.stats.errors += 1
            logger.info(
                "cache write unavailable; continuing without cache",
                extra={"error_type": type(exc).__name__},
            )
            return False


_cache: JsonCache | None = None


def get_cache(settings: Settings | None = None) -> JsonCache:
    """Return the process adapter; no network client is built unless explicitly enabled."""
    global _cache
    if _cache is None:
        config = settings or Settings.from_environment()
        if not config.cache_enabled or not config.redis_url:
            _cache = JsonCache(enabled=False)
        else:
            try:
                import redis

                client = redis.Redis.from_url(
                    config.redis_url,
                    decode_responses=False,
                    socket_connect_timeout=0.2,
                    socket_timeout=0.2,
                    health_check_interval=30,
                )
                _cache = JsonCache(client)
            except Exception as exc:
                logger.info("cache client unavailable", extra={"error_type": type(exc).__name__})
                _cache = JsonCache(enabled=False)
    return _cache


def reset_cache_for_tests() -> None:
    global _cache
    _cache = None
