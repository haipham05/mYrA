"""Bounded, HTTPS-only download for an explicitly approved open-access candidate."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from app.schemas.discovery import CatalogCandidate

MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_REDIRECTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 20.0
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}
_SCHOLARLY_HOST_SUFFIXES = (
    "acm.org",
    "annualreviews.org",
    "aps.org",
    "arxiv.org",
    "biorxiv.org",
    "cambridge.org",
    "cell.com",
    "elsevier.com",
    "europepmc.org",
    "frontiersin.org",
    "ieee.org",
    "iop.org",
    "mdpi.com",
    "medrxiv.org",
    "nature.com",
    "ncbi.nlm.nih.gov",
    "oup.com",
    "pnas.org",
    "plos.org",
    "royalsocietypublishing.org",
    "sagepub.com",
    "sciencedirect.com",
    "springer.com",
    "tandfonline.com",
    "wiley.com",
)


class ImportDownloadError(ValueError):
    """A safe import failure that does not include response bodies or credentials."""


@dataclass(frozen=True, slots=True)
class DownloadedPdf:
    data: bytes
    final_url: str
    sha256: str


def _resolve_public_addresses(host: str) -> list[str]:
    try:
        return [item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)]
    except OSError as exc:
        raise ImportDownloadError("download host could not be resolved") from exc


def _validate_https_public_url(url: str, resolve_host: Callable[[str], list[str]]) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ImportDownloadError("download URL is invalid") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or port not in (None, 443)
    ):
        raise ImportDownloadError("download URL must be a public HTTPS URL")

    hostname = parsed.hostname.rstrip(".").casefold()
    if not any(
        hostname == suffix or hostname.endswith(f".{suffix}") for suffix in _SCHOLARLY_HOST_SUFFIXES
    ):
        raise ImportDownloadError("download host is not an approved scholarly source")

    try:
        addresses = resolve_host(hostname)
    except ImportDownloadError:
        raise
    except Exception as exc:
        raise ImportDownloadError("download host could not be resolved") from exc
    if not addresses:
        raise ImportDownloadError("download host has no address")
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ImportDownloadError("download host resolved to an invalid address") from exc
        if not ip.is_global:
            raise ImportDownloadError("download host resolved to a non-public address")


async def download_open_access_pdf(
    candidate: CatalogCandidate,
    *,
    client: httpx.AsyncClient | None = None,
    resolve_host: Callable[[str], list[str]] = _resolve_public_addresses,
    max_bytes: int = MAX_PDF_BYTES,
) -> DownloadedPdf:
    """Download only the exact public PDF URL in an approved OA candidate snapshot."""
    if candidate.open_access is not True or not candidate.pdf_url:
        raise ImportDownloadError("candidate has no confirmed open-access PDF URL")
    if max_bytes < 5 or max_bytes > MAX_PDF_BYTES:
        raise ValueError("max_bytes is outside the supported PDF limit")

    owns_client = client is None
    client = client or httpx.AsyncClient(
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
        follow_redirects=False,
        trust_env=False,
    )
    current_url = candidate.pdf_url
    try:
        for redirect_count in range(MAX_REDIRECTS + 1):
            await asyncio.to_thread(_validate_https_public_url, current_url, resolve_host)
            async with client.stream(
                "GET",
                current_url,
                follow_redirects=False,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
                headers={"Accept": "application/pdf, application/octet-stream"},
            ) as response:
                if response.status_code in _REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location or redirect_count >= MAX_REDIRECTS:
                        raise ImportDownloadError("download redirect limit exceeded")
                    current_url = urljoin(current_url, location)
                    continue
                if not 200 <= response.status_code < 300:
                    raise ImportDownloadError("PDF source returned an unsuccessful response")
                content_type = (
                    response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                )
                if content_type not in _ALLOWED_CONTENT_TYPES:
                    raise ImportDownloadError("PDF source returned an unsupported content type")
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_length = int(content_length)
                    except ValueError as exc:
                        raise ImportDownloadError(
                            "PDF source returned an invalid content length"
                        ) from exc
                    if declared_length > max_bytes:
                        raise ImportDownloadError("PDF exceeds the download size limit")

                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        raise ImportDownloadError("PDF exceeds the download size limit")
                if not data.startswith(b"%PDF-"):
                    raise ImportDownloadError("downloaded file is not a PDF")
                raw = bytes(data)
                return DownloadedPdf(
                    data=raw,
                    final_url=current_url,
                    sha256=hashlib.sha256(raw).hexdigest(),
                )
        raise ImportDownloadError("download redirect limit exceeded")
    except ImportDownloadError:
        raise
    except httpx.HTTPError as exc:
        raise ImportDownloadError(type(exc).__name__) from None
    finally:
        if owns_client:
            await client.aclose()
