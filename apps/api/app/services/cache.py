"""Failure-isolated JSON cache used only for disposable derived results."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
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

    def info(self, section: str = "default") -> Mapping[str, Any]: ...


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
        namespace: str = "production",
    ) -> None:
        self._client = client
        self._enabled = enabled and client is not None
        self._max_entry_bytes = max_entry_bytes
        # Namespace material is never sent to Redis or logs in clear text.
        self._namespace = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:24]
        self.stats = CacheStats()
        self._stats_lock = threading.Lock()
        self._evicted_keys_start = self._read_evicted_keys()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def _redis_key(key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        namespace = hashlib.sha256(b"production").hexdigest()[:24]
        return f"myra:v1:{namespace}:{digest}"

    def _namespaced_redis_key(self, key: str) -> str:
        if self._namespace == hashlib.sha256(b"production").hexdigest()[:24]:
            return self._redis_key(key)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"myra:v1:{self._namespace}:{digest}"

    def _read_evicted_keys(self) -> int | None:
        if self._client is None:
            return None
        try:
            info = self._client.info("stats")
            value = info.get("evicted_keys")
            return int(value) if value is not None else None
        except Exception:
            return None

    def stats_snapshot(self) -> dict[str, int | None]:
        """Return safe process counters; Redis INFO is optional and best-effort."""
        with self._stats_lock:
            result: dict[str, int | None] = {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "errors": self.stats.errors,
                "evicted_keys_delta": None,
            }
        current = self._read_evicted_keys()
        if current is not None and self._evicted_keys_start is not None:
            result["evicted_keys_delta"] = max(0, current - self._evicted_keys_start)
        return result

    def _count(self, name: str) -> None:
        with self._stats_lock:
            setattr(self.stats, name, getattr(self.stats, name) + 1)

    def get(self, key: str, validator: Callable[[object], T]) -> T | None:
        if not self._enabled or self._client is None:
            self._count("misses")
            return None
        try:
            encoded = self._client.get(self._namespaced_redis_key(key))
            if encoded is None:
                self._count("misses")
                return None
            if isinstance(encoded, bytes):
                if len(encoded) > self._max_entry_bytes:
                    self._count("misses")
                    return None
                encoded = encoded.decode("utf-8")
            elif len(encoded.encode("utf-8")) > self._max_entry_bytes:
                self._count("misses")
                return None
            envelope = json.loads(encoded)
            if not isinstance(envelope, Mapping) or envelope.get("version") != 1:
                self._count("misses")
                return None
            value = validator(envelope.get("value"))
        except Exception as exc:
            self._count("errors")
            logger.info(
                "cache read unavailable; recomputing", extra={"error_type": type(exc).__name__}
            )
            return None
        self._count("hits")
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
                self._count("misses")
                return False
            self._client.set(self._namespaced_redis_key(key), encoded, ex=ttl_seconds)
            return True
        except Exception as exc:
            self._count("errors")
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
            _cache = JsonCache(enabled=False, namespace=config.cache_namespace)
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
                _cache = JsonCache(client, namespace=config.cache_namespace)
            except Exception as exc:
                logger.info("cache client unavailable", extra={"error_type": type(exc).__name__})
                _cache = JsonCache(enabled=False, namespace=config.cache_namespace)
    return _cache


def reset_cache_for_tests() -> None:
    global _cache
    _cache = None
