from contextlib import contextmanager
from uuid import uuid4

import pytest

from app.schemas.assistant import AssistantIntent, AssistantRunRequest, RouteDecision
from app.schemas.discovery import CatalogCandidate, CatalogSearchError, CatalogSearchResult
from app.services import assistant_tools
from app.services.assistant_tools import (
    ToolContext,
    ToolStatus,
    build_tool_registry,
    execute_tool,
    validate_tool_input,
)


def _candidate(catalog: str, catalog_id: str, **values) -> CatalogCandidate:
    return CatalogCandidate(
        catalog=catalog,
        catalog_id=catalog_id,
        title="Research on agents",
        source_url=(
            "https://arxiv.org/abs/2401.12345"
            if catalog == "arxiv"
            else "https://openalex.org/W123"
        ),
        **values,
    )


@pytest.mark.anyio
async def test_discovery_tool_returns_deduplicated_metadata_only_results(monkeypatch):
    candidate_openalex = _candidate(
        "openalex", "https://openalex.org/W123", doi="10.1234/example", publication_year=2022
    )
    candidate_arxiv = _candidate(
        "arxiv", "2401.12345", arxiv_id="2401.12345", publication_year=2022
    )

    async def search_openalex(*_args, **_kwargs):
        return CatalogSearchResult(
            catalog="openalex",
            query="agents",
            page=1,
            page_size=10,
            items=[candidate_openalex],
            has_more=False,
        )

    async def search_arxiv(*_args, **_kwargs):
        return CatalogSearchResult(
            catalog="arxiv",
            query="agents",
            page=1,
            page_size=10,
            items=[candidate_arxiv],
            has_more=False,
        )

    stages = []

    class Observation:
        def update(self, **kwargs):
            stages.append(kwargs)

    class Telemetry:
        @contextmanager
        def stage(self, name, **_kwargs):
            stages.append({"stage": name})
            yield Observation()

    monkeypatch.setattr(assistant_tools, "search_openalex", search_openalex)
    monkeypatch.setattr(assistant_tools, "search_arxiv", search_arxiv)
    monkeypatch.setattr(assistant_tools, "get_telemetry", lambda: Telemetry())

    request = AssistantRunRequest(
        message="find research about agents",
        conversation_id=uuid4(),
        project_id=uuid4(),
        idempotency_key="discovery-test-1",
    )
    decision = RouteDecision(
        intent=AssistantIntent.DISCOVER,
        action_summary="Search academic catalogs",
        arguments={"query": "agents", "year_from": 2020},
    )
    definition = build_tool_registry()[AssistantIntent.DISCOVER]
    tool_input = validate_tool_input(definition, request, decision)
    result = await execute_tool(definition, ToolContext(db=None, chat_service=None), tool_input)

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result_type == "discovery_results"
    assert result.structured_payload["metadata_only"] is True
    assert len(result.structured_payload["items"]) == 2
    assert "no papers were downloaded" in result.display_text.lower()
    assert "discovery.query" in [item.get("stage") for item in stages]
    assert "discovery.normalize" in [item.get("stage") for item in stages]


@pytest.mark.anyio
async def test_discovery_partial_catalog_failure_is_visible(monkeypatch):
    async def failed_search(*_args, **_kwargs):
        raise CatalogSearchError("openalex", "TimeoutException")

    async def empty_arxiv(*_args, **_kwargs):
        return CatalogSearchResult(
            catalog="arxiv", query="agents", page=1, page_size=10, items=[], has_more=False
        )

    monkeypatch.setattr(assistant_tools, "search_openalex", failed_search)
    monkeypatch.setattr(assistant_tools, "search_arxiv", empty_arxiv)

    request = AssistantRunRequest(
        message="find agents",
        conversation_id=uuid4(),
        project_id=uuid4(),
        idempotency_key="discovery-test-2",
    )
    decision = RouteDecision(
        intent=AssistantIntent.DISCOVER,
        action_summary="Search academic catalogs",
        arguments={"query": "agents"},
    )
    definition = build_tool_registry()[AssistantIntent.DISCOVER]
    tool_input = validate_tool_input(definition, request, decision)
    result = await execute_tool(definition, ToolContext(db=None, chat_service=None), tool_input)

    assert result.status is ToolStatus.SUCCEEDED
    assert result.structured_payload["source_errors"] == {"openalex": "TimeoutException"}
    assert any("openalex search failed" in warning for warning in result.warnings)
