import json

import httpx
import pytest

from app.schemas.discovery import CatalogSearchError
from app.services.discovery.catalogs import search_arxiv, search_openalex


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.anyio
async def test_openalex_normalizes_metadata_and_bounds_page_size():
    requested = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requested.update(dict(request.url.params))
        return httpx.Response(
            200,
            json={
                "meta": {"count": 60},
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "title": "A Study",
                        "publication_year": 2024,
                        "doi": "https://doi.org/10.1234/example",
                        "ids": {"arxiv": "https://arxiv.org/abs/2401.12345v2"},
                        "authorships": [
                            {"author": {"display_name": "Author One"}},
                        ],
                        "abstract_inverted_index": {"hello": [1], "Research": [0]},
                        "open_access": {"is_oa": True},
                        "locations": [
                            {"pdf_url": "https://example.org/paper.pdf"},
                        ],
                    }
                ],
            },
        )

    async with _client(handler) as client:
        result = await search_openalex("  study   systems ", page=2, page_size=25, client=client)

    assert requested["page"] == "2"
    assert requested["per-page"] == "25"
    assert result.has_more is True
    candidate = result.items[0]
    assert candidate.doi == "10.1234/example"
    assert candidate.arxiv_id == "2401.12345v2"
    assert candidate.abstract == "Research hello"
    assert candidate.authors == ["Author One"]
    assert candidate.open_access is True


@pytest.mark.anyio
async def test_arxiv_normalizes_atom_entry_and_pagination():
    body = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
      <opensearch:totalResults>26</opensearch:totalResults>
      <entry>
        <id>https://arxiv.org/abs/2401.54321v3</id>
        <title> A Useful Paper </title><summary> Abstract text </summary>
        <published>2023-12-01T00:00:00Z</published>
        <author><name>Researcher</name></author>
        <link title="pdf" href="https://arxiv.org/pdf/2401.54321" />
      </entry>
    </feed>"""
    requested = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requested.update(dict(request.url.params))
        return httpx.Response(200, text=body)

    async with _client(handler) as client:
        result = await search_arxiv("attention models", page=2, page_size=25, client=client)

    assert requested["start"] == "25"
    assert requested["max_results"] == "25"
    assert result.has_more is False
    assert result.items[0].arxiv_id == "2401.54321v3"
    assert result.items[0].title == "A Useful Paper"
    assert result.items[0].open_access is True


@pytest.mark.anyio
async def test_catalog_errors_are_safe_and_malformed_payload_is_not_empty_success():
    def error_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="private provider body")

    async with _client(error_handler) as client:
        with pytest.raises(CatalogSearchError) as error:
            await search_openalex("query", client=client)
    assert error.value.reason == "HTTPStatusError"
    assert "private provider body" not in str(error.value)

    def malformed_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"results": "not-a-list"}))

    async with _client(malformed_handler) as client:
        with pytest.raises(CatalogSearchError, match="invalid response"):
            await search_openalex("query", client=client)


@pytest.mark.anyio
async def test_arxiv_malformed_xml_is_a_safe_provider_error():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<feed>")

    async with _client(handler) as client:
        with pytest.raises(CatalogSearchError) as error:
            await search_arxiv("query", client=client)
    assert error.value.catalog == "arxiv"
    assert error.value.reason == "ParseError"


@pytest.mark.parametrize(
    ("query", "page", "page_size"),
    [("", 1, 10), ("x" * 501, 1, 10), ("query", 4, 10), ("query", 1, 26)],
)
def test_search_inputs_are_bounded(query: str, page: int, page_size: int):
    with pytest.raises(ValueError):
        from app.services.discovery.catalogs import _bounded_query

        _bounded_query(query, page, page_size)


def test_openalex_ignores_untrusted_work_ids_and_bounds_abstracts():
    from app.services.discovery.catalogs import _abstract_from_inverted_index, _openalex_candidate

    item = {"id": "https://attacker.invalid/W1", "title": "Unexpected source"}
    assert _openalex_candidate(item) is None
    long_index = {"word": list(range(50_100))}
    assert len(_abstract_from_inverted_index(long_index) or "") <= 50_000
