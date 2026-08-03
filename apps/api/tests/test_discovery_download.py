import hashlib

import httpx
import pytest

from app.schemas.discovery import CatalogCandidate
from app.services.discovery.download import ImportDownloadError, download_open_access_pdf


def _candidate(url: str = "https://repo.example/paper.pdf", **values) -> CatalogCandidate:
    candidate = {
        "catalog": "arxiv",
        "catalog_id": "2401.12345",
        "title": "Open Access Paper",
        "source_url": "https://arxiv.org/abs/2401.12345",
        "pdf_url": url,
        "open_access": True,
    }
    candidate.update(values)
    return CatalogCandidate(**candidate)


def _public_resolver(_host: str) -> list[str]:
    return ["93.184.216.34"]


@pytest.mark.anyio
async def test_downloads_and_hashes_a_bounded_pdf():
    pdf = b"%PDF-1.7\nsmall fixture"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "repo.example"
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=pdf)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await download_open_access_pdf(
            _candidate(), client=client, resolve_host=_public_resolver
        )

    assert result.data == pdf
    assert result.sha256 == hashlib.sha256(pdf).hexdigest()
    assert result.final_url == "https://repo.example/paper.pdf"


@pytest.mark.anyio
async def test_private_redirect_is_rejected_before_following_it():
    requested_hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

    def resolver(host: str) -> list[str]:
        return ["93.184.216.34"] if host == "repo.example" else ["127.0.0.1"]

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ImportDownloadError, match="public HTTPS"):
            await download_open_access_pdf(_candidate(), client=client, resolve_host=resolver)

    assert requested_hosts == ["repo.example"]


@pytest.mark.anyio
async def test_private_dns_destination_is_rejected_before_request():
    requested = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requested
        requested = True
        return httpx.Response(200, content=b"%PDF-test")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ImportDownloadError, match="non-public"):
            await download_open_access_pdf(
                _candidate(), client=client, resolve_host=lambda _host: ["10.0.0.3"]
            )
    assert requested is False


@pytest.mark.anyio
async def test_oversized_and_non_pdf_responses_are_rejected():
    async def oversized(*, content_type: str, body: bytes, max_bytes: int):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": content_type}, content=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await download_open_access_pdf(
                _candidate(),
                client=client,
                resolve_host=_public_resolver,
                max_bytes=max_bytes,
            )

    with pytest.raises(ImportDownloadError, match="size limit"):
        await oversized(content_type="application/pdf", body=b"%PDF-" + b"x" * 100, max_bytes=50)
    with pytest.raises(ImportDownloadError, match="content type"):
        await oversized(content_type="text/html", body=b"%PDF-not-really", max_bytes=100)
    with pytest.raises(ImportDownloadError, match="not a PDF"):
        await oversized(content_type="application/octet-stream", body=b"not a pdf", max_bytes=100)


@pytest.mark.anyio
async def test_non_open_access_or_missing_pdf_is_not_downloaded():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    ) as client:
        with pytest.raises(ImportDownloadError, match="confirmed open-access"):
            await download_open_access_pdf(
                _candidate(open_access=False), client=client, resolve_host=_public_resolver
            )
        with pytest.raises(ImportDownloadError, match="confirmed open-access"):
            await download_open_access_pdf(
                _candidate(pdf_url=None), client=client, resolve_host=_public_resolver
            )
