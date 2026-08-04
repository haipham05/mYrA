from __future__ import annotations

import asyncio
import json

from app.schemas.discovery import CatalogCandidate, CatalogSearchResult
from app.services import assistant_tools
from app.services.cache import JsonCache


class MemoryRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.fail = False

    def info(self, _section: str = "default") -> dict[str, int]:
        return {"evicted_keys": 0}

    def get(self, name: str) -> str | None:
        if self.fail:
            raise OSError("offline")
        return self.values.get(name)

    def set(self, name: str, value: str, *, ex: int) -> None:
        if self.fail:
            raise OSError("offline")
        assert ex == 600
        self.values[name] = value


def search_result(catalog: str = "openalex", query: str = "attention") -> CatalogSearchResult:
    return CatalogSearchResult(
        catalog=catalog,
        query=query,
        page=1,
        page_size=10,
        items=[
            CatalogCandidate(
                catalog=catalog,
                catalog_id=f"{catalog}-1",
                title="Attention",
                source_url=f"https://{catalog}.org/1",
            )
        ],
        has_more=False,
    )


def test_catalog_cache_reuses_public_metadata(monkeypatch):
    redis = MemoryRedis()
    cache = JsonCache(redis)
    calls = 0

    async def search(_query: str, *, page_size: int):
        nonlocal calls
        assert page_size == 10
        calls += 1
        return search_result()

    monkeypatch.setattr(assistant_tools, "get_cache", lambda: cache)
    monkeypatch.setattr(assistant_tools, "search_openalex", search)

    first, first_status = asyncio.run(
        assistant_tools._cached_catalog_search("openalex", "attention")
    )
    second, second_status = asyncio.run(
        assistant_tools._cached_catalog_search("openalex", "attention")
    )

    assert first_status == "miss"
    assert second_status == "hit"
    assert first == second
    assert calls == 1


def test_invalid_catalog_cache_entry_is_refetched(monkeypatch):
    redis = MemoryRedis()
    cache = JsonCache(redis)
    redis.values[cache._namespaced_redis_key("discovery:v1:openalex:attention:1:10")] = json.dumps(
        {"version": 1, "value": search_result("arxiv").model_dump(mode="json")}
    )
    calls = 0

    async def search(_query: str, *, page_size: int):
        nonlocal calls
        calls += 1
        return search_result()

    monkeypatch.setattr(assistant_tools, "get_cache", lambda: cache)
    monkeypatch.setattr(assistant_tools, "search_openalex", search)

    result, status = asyncio.run(assistant_tools._cached_catalog_search("openalex", "attention"))

    assert result.catalog == "openalex"
    assert status == "miss"
    assert calls == 1


def test_cache_outage_does_not_hide_catalog_result(monkeypatch):
    redis = MemoryRedis()
    redis.fail = True
    cache = JsonCache(redis)

    async def search(_query: str, *, page_size: int):
        assert page_size == 10
        return search_result()

    monkeypatch.setattr(assistant_tools, "get_cache", lambda: cache)
    monkeypatch.setattr(assistant_tools, "search_openalex", search)

    result, status = asyncio.run(assistant_tools._cached_catalog_search("openalex", "attention"))

    assert result.items[0].title == "Attention"
    assert status == "error_recomputed"
