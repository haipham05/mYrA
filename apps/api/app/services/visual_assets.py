"""Create bounded, source-bound visual crops from stored research PDFs."""

from __future__ import annotations

import asyncio
import hashlib
import io
import math
from dataclasses import dataclass
from typing import TypedDict
from uuid import UUID

import pypdfium2 as pdfium
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Paper, PaperElement
from app.storage import ObjectStorage

MAX_SOURCE_BYTES = 50 * 1024 * 1024
MAX_RENDER_DIMENSION = 1600
MAX_RENDER_PIXELS = 2_000_000
MAX_PNG_BYTES = 4 * 1024 * 1024
DEFAULT_RENDER_SCALE = 2.0
_SHA256_LENGTH = 64


class VisualAssetError(ValueError):
    """Safe validation or rendering failure while building a visual source."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class NormalizedCropBox:
    """Crop bounds as fractions of a page, with a top-left origin."""

    left: float
    top: float
    right: float
    bottom: float

    def validate(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(value) for value in values):
            raise VisualAssetError("INVALID_GEOMETRY", "Crop coordinates must be finite.")
        if not (0 <= self.left < self.right <= 1 and 0 <= self.top < self.bottom <= 1):
            raise VisualAssetError(
                "INVALID_GEOMETRY",
                "Crop bounds must be ordered and within the normalized page.",
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
        }


@dataclass(frozen=True, slots=True)
class VisualAsset:
    """Ephemeral PNG crop and primitive provenance; bytes are not persisted."""

    png_bytes: bytes
    source: VisualSourceMetadata


class VisualSourceMetadata(TypedDict):
    project_id: str
    paper_id: str
    filename: str
    document_sha256: str
    page_number: int
    page_width: float
    page_height: float
    crop_box_normalized_top_left: dict[str, float]
    crop_sha256: str
    mime_type: str
    byte_length: int
    pixel_width: int
    pixel_height: int
    caption: str | None
    text_citation: bool


def _storage_key(storage_path: str, storage: ObjectStorage) -> str:
    if storage_path.startswith("gs://"):
        parts = storage_path.split("/", 3)
        if len(parts) != 4 or not parts[3]:
            raise VisualAssetError("INVALID_SOURCE", "The stored PDF location is invalid.")
        bucket_name = getattr(storage, "bucket_name", None)
        if bucket_name and parts[2] != bucket_name:
            raise VisualAssetError("INVALID_SOURCE", "The PDF belongs to another storage bucket.")
        return parts[3]
    if storage_path.startswith("memory://"):
        return storage_path.removeprefix("memory://")
    return storage_path


async def _read_source_with_limit(storage: ObjectStorage, key: str) -> bytes:
    collected = bytearray()
    async for chunk in storage.open_stream(key, chunk_size=64 * 1024):
        if len(collected) + len(chunk) > MAX_SOURCE_BYTES:
            raise VisualAssetError(
                "SOURCE_TOO_LARGE", "The original PDF exceeds the extraction limit."
            )
        collected.extend(chunk)
    return bytes(collected)


def _render_png(
    pdf_bytes: bytes, page_number: int, crop: NormalizedCropBox
) -> tuple[bytes, int, int, float, float]:
    try:
        document = pdfium.PdfDocument(pdf_bytes)
        if page_number < 1 or page_number > len(document):
            raise VisualAssetError("PAGE_OUT_OF_RANGE", "The selected page is outside the PDF.")
        page = document[page_number - 1]
        try:
            width, height = page.get_size()
            if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
                raise VisualAssetError(
                    "INVALID_PAGE", "The selected PDF page has invalid dimensions."
                )

            crop_width = (crop.right - crop.left) * width
            crop_height = (crop.bottom - crop.top) * height
            scale = min(
                DEFAULT_RENDER_SCALE,
                MAX_RENDER_DIMENSION / crop_width,
                MAX_RENDER_DIMENSION / crop_height,
                math.sqrt(MAX_RENDER_PIXELS / (crop_width * crop_height)),
            )
            # PDFium crop values are edge cutoffs in (left, bottom, right, top) order.
            bitmap = page.render(
                scale=scale,
                crop=(
                    crop.left * width,
                    (1 - crop.bottom) * height,
                    (1 - crop.right) * width,
                    crop.top * height,
                ),
                rev_byteorder=True,
            )
            image = bitmap.to_pil().convert("RGB")
            try:
                if (
                    image.width > MAX_RENDER_DIMENSION
                    or image.height > MAX_RENDER_DIMENSION
                    or image.width * image.height > MAX_RENDER_PIXELS
                ):
                    raise VisualAssetError(
                        "RENDER_TOO_LARGE", "The selected crop exceeds render limits."
                    )
                output = io.BytesIO()
                image.save(output, format="PNG", optimize=True)
                png_bytes = output.getvalue()
                if len(png_bytes) > MAX_PNG_BYTES:
                    raise VisualAssetError(
                        "OUTPUT_TOO_LARGE", "The rendered crop exceeds the size limit."
                    )
                return png_bytes, image.width, image.height, width, height
            finally:
                image.close()
                bitmap.close()
        finally:
            page.close()
            document.close()
    except VisualAssetError:
        raise
    except Exception as err:
        raise VisualAssetError(
            "PDF_RENDER_FAILED", "The selected PDF page could not be rendered."
        ) from err


async def extract_visual_asset(
    *,
    session: Session,
    storage: ObjectStorage,
    project_id: UUID,
    paper_id: UUID,
    page_number: int,
    crop: NormalizedCropBox,
    caption_element_id: UUID | None = None,
) -> VisualAsset:
    """Render a bounded crop after verifying its project, READY paper, and source hash.

    The returned bytes remain in process memory for a later caller. This helper does not
    persist assets or send them to a provider.
    """
    crop.validate()
    if page_number < 1:
        raise VisualAssetError("PAGE_OUT_OF_RANGE", "Page numbers start at one.")

    paper = session.scalar(
        select(Paper).where(Paper.id == paper_id, Paper.project_id == project_id)
    )
    if paper is None:
        raise VisualAssetError(
            "PAPER_NOT_FOUND", "The selected paper was not found in this project."
        )
    if paper.status != "READY":
        raise VisualAssetError("PAPER_NOT_READY", "Visual extraction requires a READY paper.")
    if not paper.document_sha256 or len(paper.document_sha256) != _SHA256_LENGTH:
        raise VisualAssetError("SOURCE_HASH_MISSING", "The paper has no valid source hash.")
    try:
        int(paper.document_sha256, 16)
    except ValueError as err:
        raise VisualAssetError(
            "SOURCE_HASH_MISSING", "The paper has no valid source hash."
        ) from err
    if paper.page_count is not None and page_number > paper.page_count:
        raise VisualAssetError("PAGE_OUT_OF_RANGE", "The selected page is outside the paper.")

    caption: str | None = None
    if caption_element_id is not None:
        element = session.scalar(
            select(PaperElement).where(
                PaperElement.id == caption_element_id,
                PaperElement.paper_id == paper.id,
                PaperElement.page_number == page_number,
                PaperElement.element_type.in_(("caption", "figure_caption", "table_caption")),
            )
        )
        if element is None:
            raise VisualAssetError(
                "CAPTION_NOT_FOUND", "The caption element is not on the selected paper page."
            )
        caption = element.text

    try:
        pdf_bytes = await _read_source_with_limit(
            storage, _storage_key(paper.storage_path, storage)
        )
    except VisualAssetError:
        raise
    except Exception as err:
        raise VisualAssetError(
            "SOURCE_UNAVAILABLE", "The original PDF could not be loaded."
        ) from err
    source_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    if source_sha256 != paper.document_sha256.lower():
        raise VisualAssetError(
            "SOURCE_HASH_MISMATCH", "The stored PDF no longer matches this paper."
        )

    png_bytes, pixel_width, pixel_height, page_width, page_height = await asyncio.to_thread(
        _render_png, pdf_bytes, page_number, crop
    )
    source: VisualSourceMetadata = {
        "project_id": str(project_id),
        "paper_id": str(paper.id),
        "filename": paper.filename,
        "document_sha256": source_sha256,
        "page_number": page_number,
        "page_width": page_width,
        "page_height": page_height,
        "crop_box_normalized_top_left": crop.as_dict(),
        "crop_sha256": hashlib.sha256(png_bytes).hexdigest(),
        "mime_type": "image/png",
        "byte_length": len(png_bytes),
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "caption": caption,
        "text_citation": False,
    }
    return VisualAsset(png_bytes=png_bytes, source=source)
