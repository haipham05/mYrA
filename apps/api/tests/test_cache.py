import json

from app.config import Settings
from app.services.cache import JsonCache, get_cache, reset_cache_for_tests


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.error: Exception | None = None

    def get(self, name: str) -> str | None:
        if self.error:
            raise self.error
        return self.values.get(name)

    def set(self, name: str, value: str, *, ex: int) -> bool:
        if self.error:
            raise self.error
        self.values[name] = value
        self.ttls[name] = ex
        return True


def int_value(value: object) -> int:
    if not isinstance(value, int):
        raise ValueError("not an integer")
    return value


def test_json_cache_round_trip_hashes_keys_and_applies_ttl() -> None:
    client = FakeRedis()
    cache = JsonCache(client)

    assert cache.set("private query text", {"score": 3}, ttl_seconds=60)
    redis_key = next(iter(client.values))
    assert "private query text" not in redis_key
    assert client.ttls[redis_key] == 60
    assert cache.get("private query text", lambda value: value) == {"score": 3}
    assert cache.stats.hits == 1


def test_disabled_cache_misses_without_network_access() -> None:
    cache = JsonCache(enabled=False)
    assert cache.get("key", int_value) is None
    assert not cache.set("key", 1, ttl_seconds=60)
    assert cache.stats.misses == 1


def test_cache_rejects_malformed_wrong_shape_oversized_and_non_json_values() -> None:
    client = FakeRedis()
    cache = JsonCache(client, max_entry_bytes=64)
    key = cache._redis_key("key")
    client.values[key] = "not-json"
    assert cache.get("key", int_value) is None

    client.values[key] = json.dumps({"version": 1, "value": "wrong"})
    assert cache.get("key", int_value) is None

    client.values[key] = " " * 65
    assert cache.get("key", int_value) is None
    assert not cache.set("key", object(), ttl_seconds=60)
    assert not cache.set("key", "x" * 100, ttl_seconds=60)


def test_redis_outage_is_a_miss_or_ignored_write() -> None:
    client = FakeRedis()
    cache = JsonCache(client)
    client.error = TimeoutError("private endpoint")
    assert cache.get("key", int_value) is None
    assert not cache.set("key", 1, ttl_seconds=60)
    assert cache.stats.errors == 2


def test_invalid_ttl_is_not_written() -> None:
    client = FakeRedis()
    cache = JsonCache(client)
    assert not cache.set("key", 1, ttl_seconds=0)
    assert client.values == {}


def test_reset_cache_for_tests_is_safe() -> None:
    reset_cache_for_tests()


def test_factory_is_disabled_by_default_and_explicitly_configured() -> None:
    reset_cache_for_tests()
    assert not get_cache(Settings()).enabled
    reset_cache_for_tests()
    configured = get_cache(Settings(cache_enabled=True, redis_url="redis://localhost:6379/0"))
    assert configured.enabled
    reset_cache_for_tests()
