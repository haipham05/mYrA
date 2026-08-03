"""Bounded OpenAlex and arXiv search with no import or database side effects."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from app.schemas.discovery import (
    CatalogCandidate,
    CatalogSearchError,
    CatalogSearchResult,
)

OPENALEX_URL = "https://api.openalex.org/works"
ARXIV_URL = "https://export.arxiv.org/api/query"
MAX_PAGE = 3
MAX_PAGE_SIZE = 25
TIMEOUT_SECONDS = 10.0
_ARXIV_ID = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", re.I)
_OPENALEX_WORK_ID = re.compile(r"^https://openalex\.org/W[0-9]+$", re.I)


def _bounded_query(query: str, page: int, page_size: int) -> tuple[str, int, int]:
    clean_query = " ".join(query.split())
    if not clean_query or len(clean_query) > 500:
        raise ValueError("query must contain between 1 and 500 characters")
    if not 1 <= page <= MAX_PAGE:
        raise ValueError(f"page must be between 1 and {MAX_PAGE}")
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}")
    return clean_query, page, page_size


def _abstract_from_inverted_index(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    words: dict[int, str] = {}
    for token, positions in value.items():
        if isinstance(token, str) and isinstance(positions, list):
            for position in positions:
                if isinstance(position, int) and 0 <= position < 100_000:
                    words[position] = token
    return " ".join(words[index] for index in sorted(words))[:50_000] or None


def _openalex_candidate(item: dict[str, Any]) -> CatalogCandidate | None:
    work_id = item.get("id")
    title = item.get("title") or item.get("display_name")
    if (
        not isinstance(work_id, str)
        or not _OPENALEX_WORK_ID.fullmatch(work_id)
        or not isinstance(title, str)
        or not title.strip()
    ):
        return None

    authorships = item.get("authorships")
    authors = []
    if isinstance(authorships, list):
        for authorship in authorships[:100]:
            author = authorship.get("author") if isinstance(authorship, dict) else None
            name = author.get("display_name") if isinstance(author, dict) else None
            if isinstance(name, str) and name.strip():
                authors.append(name.strip()[:500])

    doi = item.get("doi")
    doi = doi.removeprefix("https://doi.org/") if isinstance(doi, str) else None
    locations = item.get("locations")
    pdf_url = None
    if isinstance(locations, list):
        for location in locations:
            if not isinstance(location, dict):
                continue
            landing = location.get("landing_page_url")
            pdf = location.get("pdf_url")
            if isinstance(pdf, str) and pdf.startswith("https://"):
                pdf_url = pdf[:4_000]
                break
            if isinstance(landing, str) and landing.startswith("https://"):
                pdf_url = pdf_url or None

    ids = item.get("ids")
    arxiv_id = None
    if isinstance(ids, dict) and isinstance(ids.get("arxiv"), str):
        match = _ARXIV_ID.search(ids["arxiv"])
        arxiv_id = match.group(1) if match else None

    year = item.get("publication_year")
    open_access = item.get("open_access")
    open_access_value = open_access.get("is_oa") if isinstance(open_access, dict) else None
    return CatalogCandidate(
        catalog="openalex",
        catalog_id=work_id[:300],
        title=title.strip()[:2_000],
        authors=authors,
        publication_year=year if isinstance(year, int) and 1000 <= year <= 2100 else None,
        doi=doi[:255] if isinstance(doi, str) and doi else None,
        arxiv_id=arxiv_id,
        abstract=_abstract_from_inverted_index(item.get("abstract_inverted_index")),
        source_url=work_id[:4_000],
        pdf_url=pdf_url,
        open_access=open_access_value if isinstance(open_access_value, bool) else None,
    )


def _arxiv_candidate(entry: ET.Element, namespace: str) -> CatalogCandidate | None:
    def content(name: str) -> str | None:
        node = entry.find(f"{{{namespace}}}{name}")
        return node.text.strip() if node is not None and node.text else None

    identifier = content("id")
    title = content("title")
    if not identifier or not title:
        return None
    match = _ARXIV_ID.search(identifier)
    if not match:
        return None

    author_nodes = entry.findall(f"{{{namespace}}}author")
    authors = [
        name.text.strip()[:500]
        for author in author_nodes[:100]
        if (name := author.find(f"{{{namespace}}}name")) is not None
        and name.text
        and name.text.strip()
    ]
    year = None
    published = content("published")
    if published and len(published) >= 4 and published[:4].isdigit():
        year = int(published[:4])
    pdf_url = None
    for link in entry.findall(f"{{{namespace}}}link"):
        if link.attrib.get("title") == "pdf" and link.attrib.get("href", "").startswith("https://"):
            pdf_url = link.attrib["href"][:4_000]
            break
    return CatalogCandidate(
        catalog="arxiv",
        catalog_id=match.group(1),
        title=title.replace("\n", " ").strip()[:2_000],
        authors=authors,
        publication_year=year,
        arxiv_id=match.group(1),
        abstract=(content("summary") or "")[:50_000] or None,
        source_url=f"https://arxiv.org/abs/{match.group(1)}",
        pdf_url=pdf_url,
        open_access=True,
    )


async def search_openalex(
    query: str,
    *,
    page: int = 1,
    page_size: int = 10,
    client: httpx.AsyncClient | None = None,
) -> CatalogSearchResult:
    clean_query, page, page_size = _bounded_query(query, page, page_size)
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False)
    try:
        response = await client.get(
            OPENALEX_URL,
            params={"search": clean_query, "page": page, "per-page": page_size},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise CatalogSearchError("openalex", type(exc).__name__) from None
    finally:
        if owns_client:
            await client.aclose()

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise CatalogSearchError("openalex", "invalid response")
    raw_items = payload["results"][:page_size]
    items = [
        candidate
        for item in raw_items
        if isinstance(item, dict)
        if (candidate := _openalex_candidate(item))
    ]
    count = (
        (payload.get("meta") or {}).get("count") if isinstance(payload.get("meta"), dict) else None
    )
    has_more = len(items) >= page_size or (isinstance(count, int) and count > page * page_size)
    return CatalogSearchResult(
        catalog="openalex",
        query=clean_query,
        page=page,
        page_size=page_size,
        items=items,
        has_more=has_more,
    )


async def search_arxiv(
    query: str,
    *,
    page: int = 1,
    page_size: int = 10,
    client: httpx.AsyncClient | None = None,
) -> CatalogSearchResult:
    clean_query, page, page_size = _bounded_query(query, page, page_size)
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False)
    try:
        response = await client.get(
            ARXIV_URL,
            params={
                "search_query": f"all:{clean_query}",
                "start": (page - 1) * page_size,
                "max_results": page_size,
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (httpx.HTTPError, ET.ParseError, ValueError) as exc:
        raise CatalogSearchError("arxiv", type(exc).__name__) from None
    finally:
        if owns_client:
            await client.aclose()

    atom_namespace = "http://www.w3.org/2005/Atom"
    entries = root.findall(f"{{{atom_namespace}}}entry")[:page_size]
    items = [
        candidate for entry in entries if (candidate := _arxiv_candidate(entry, atom_namespace))
    ]
    total = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    count = int(total.text) if total is not None and total.text and total.text.isdigit() else None
    return CatalogSearchResult(
        catalog="arxiv",
        query=clean_query,
        page=page,
        page_size=page_size,
        items=items,
        has_more=isinstance(count, int) and count > page * page_size,
    )
