import json

from app.config import Settings
from app.services.cache import JsonCache, get_cache, reset_cache_for_tests


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.error: Exception | None = None
        self.evicted_keys = 0

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

    def info(self, section: str = "default") -> dict[str, int]:
        if self.error:
            raise self.error
        return {"evicted_keys": self.evicted_keys}


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


def test_namespace_is_hashed_and_isolates_entries_without_deleting_shared_data() -> None:
    client = FakeRedis()
    production = JsonCache(client)
    benchmark = JsonCache(client, namespace="benchmark-private-run")
    assert production.set("same-key", 1, ttl_seconds=60)
    assert benchmark.get("same-key", int_value) is None
    assert benchmark.set("same-key", 2, ttl_seconds=60)
    assert production.get("same-key", int_value) == 1
    assert benchmark.get("same-key", int_value) == 2
    assert len(client.values) == 2
    assert all("benchmark-private-run" not in key for key in client.values)


def test_stats_snapshot_tracks_counters_and_best_effort_redis_evictions() -> None:
    client = FakeRedis()
    cache = JsonCache(client)
    before = cache.stats_snapshot()
    cache.get("missing", int_value)
    client.evicted_keys += 3
    after = cache.stats_snapshot()
    assert after == {"hits": 0, "misses": 1, "errors": 0, "evicted_keys_delta": 3}
    assert before["evicted_keys_delta"] == 0


def test_stats_failure_does_not_change_cache_product_behavior() -> None:
    client = FakeRedis()
    cache = JsonCache(client)
    assert cache.set("key", 7, ttl_seconds=60)
    client.error = RuntimeError("private redis endpoint")
    assert cache.get("key", int_value) is None
    snapshot = cache.stats_snapshot()
    assert snapshot["errors"] == 1
    assert snapshot["evicted_keys_delta"] is None


def test_reset_cache_for_tests_is_safe() -> None:
    reset_cache_for_tests()


def test_factory_is_disabled_by_default_and_explicitly_configured() -> None:
    reset_cache_for_tests()
    assert not get_cache(Settings()).enabled
    reset_cache_for_tests()
    configured = get_cache(Settings(cache_enabled=True, redis_url="redis://localhost:6379/0"))
    assert configured.enabled
    reset_cache_for_tests()


def test_cache_namespace_is_read_only_from_explicit_environment(monkeypatch) -> None:
    monkeypatch.setenv("MYRA_CACHE_NAMESPACE", "benchmark-test")
    assert Settings.from_environment().cache_namespace == "benchmark-test"
