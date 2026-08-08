from __future__ import annotations

import asyncio
import hashlib
import io
import math
import struct

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import Paper, PaperElement, Project
from app.services import visual_assets
from app.services.visual_assets import (
    NormalizedCropBox,
    VisualAssetError,
    extract_visual_asset,
)
from app.storage.local import MemoryStorage


def _pdf_bytes() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=100)
    content = DecodedStreamObject()
    content.set_data(b"1 0 0 rg 40 20 40 40 re f\n0 0 1 rg 80 60 30 20 re f\n")
    page[NameObject("/Contents")] = writer._add_object(content)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.fixture
def research_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'visual-assets.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as session:
        project = Project(name="Visual source test")
        session.add(project)
        session.flush()
        pdf = _pdf_bytes()
        storage = MemoryStorage()
        storage_path = "papers/source.pdf"
        paper = Paper(
            project_id=project.id,
            filename="source.pdf",
            storage_path=storage_path,
            document_sha256=hashlib.sha256(pdf).hexdigest(),
            status="READY",
            page_count=1,
        )
        session.add(paper)
        session.flush()
        caption = PaperElement(
            paper_id=paper.id,
            page_number=1,
            element_index=0,
            element_type="caption",
            text="Figure 1: Example plot.",
        )
        session.add(caption)
        session.commit()
        # MemoryStorage's get key is the object key independent of its returned URI.
        asyncio.run(storage.put(storage_path, pdf))
        yield session, project, paper, caption, storage, pdf
    engine.dispose()


def test_extracts_bounded_crop_with_source_metadata(research_db):
    session, project, paper, caption, storage, pdf = research_db

    asset = asyncio.run(
        extract_visual_asset(
            session=session,
            storage=storage,
            project_id=project.id,
            paper_id=paper.id,
            page_number=1,
            crop=NormalizedCropBox(left=0.1, top=0.2, right=0.6, bottom=0.7),
            caption_element_id=caption.id,
        )
    )

    assert asset.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert asset.source["project_id"] == str(project.id)
    assert asset.source["paper_id"] == str(paper.id)
    assert asset.source["document_sha256"] == hashlib.sha256(pdf).hexdigest()
    assert asset.source["crop_sha256"] == hashlib.sha256(asset.png_bytes).hexdigest()
    assert asset.source["page_number"] == 1
    assert asset.source["caption"] == caption.text
    assert asset.source["text_citation"] is False
    assert asset.source["pixel_width"] <= visual_assets.MAX_RENDER_DIMENSION
    assert asset.source["pixel_height"] <= visual_assets.MAX_RENDER_DIMENSION
    assert (
        asset.source["pixel_width"] * asset.source["pixel_height"]
        <= visual_assets.MAX_RENDER_PIXELS
    )
    assert asset.source["byte_length"] == len(asset.png_bytes)
    assert asset.source["byte_length"] <= visual_assets.MAX_PNG_BYTES
    assert asset.source["crop_box_normalized_top_left"] == {
        "left": 0.1,
        "top": 0.2,
        "right": 0.6,
        "bottom": 0.7,
    }
    assert struct.unpack(">II", asset.png_bytes[16:24]) == (200, 99)
    with Image.open(io.BytesIO(asset.png_bytes)) as image:
        image = image.convert("RGB")
        red, green, blue = image.getpixel((40, 40))
        assert red > 200 and green < 40 and blue < 40
        red, green, blue = image.getpixel((150, 20))
        assert red < 40 and green < 40 and blue > 200
        assert image.getpixel((40, 10)) == (255, 255, 255)


def test_rejects_wrong_project_and_unready_paper_before_storage_read(research_db):
    session, project, paper, _, storage, _ = research_db
    other_project = Project(name="Different project")
    session.add(other_project)
    session.commit()

    with pytest.raises(VisualAssetError, match="not found") as error:
        asyncio.run(
            extract_visual_asset(
                session=session,
                storage=storage,
                project_id=other_project.id,
                paper_id=paper.id,
                page_number=1,
                crop=NormalizedCropBox(0, 0, 1, 1),
            )
        )
    assert error.value.code == "PAPER_NOT_FOUND"

    paper.status = "PROCESSING"
    session.commit()
    with pytest.raises(VisualAssetError) as error:
        asyncio.run(
            extract_visual_asset(
                session=session,
                storage=storage,
                project_id=project.id,
                paper_id=paper.id,
                page_number=1,
                crop=NormalizedCropBox(0, 0, 1, 1),
            )
        )
    assert error.value.code == "PAPER_NOT_READY"


def test_rejects_changed_source_hash(research_db):
    session, project, paper, _, storage, _ = research_db
    asyncio.run(storage.put(paper.storage_path, b"not the original pdf"))

    with pytest.raises(VisualAssetError) as error:
        asyncio.run(
            extract_visual_asset(
                session=session,
                storage=storage,
                project_id=project.id,
                paper_id=paper.id,
                page_number=1,
                crop=NormalizedCropBox(0, 0, 1, 1),
            )
        )
    assert error.value.code == "SOURCE_HASH_MISMATCH"


def test_rejects_page_outside_document_and_unassociated_caption(research_db):
    session, project, paper, caption, storage, _ = research_db
    with pytest.raises(VisualAssetError) as error:
        asyncio.run(
            extract_visual_asset(
                session=session,
                storage=storage,
                project_id=project.id,
                paper_id=paper.id,
                page_number=2,
                crop=NormalizedCropBox(0, 0, 1, 1),
            )
        )
    assert error.value.code == "PAGE_OUT_OF_RANGE"

    foreign_caption = PaperElement(
        paper_id=paper.id,
        page_number=2,
        element_index=1,
        element_type="caption",
        text="Foreign page caption",
    )
    session.add(foreign_caption)
    session.commit()
    with pytest.raises(VisualAssetError) as error:
        asyncio.run(
            extract_visual_asset(
                session=session,
                storage=storage,
                project_id=project.id,
                paper_id=paper.id,
                page_number=1,
                crop=NormalizedCropBox(0, 0, 1, 1),
                caption_element_id=foreign_caption.id,
            )
        )
    assert error.value.code == "CAPTION_NOT_FOUND"
    assert caption.text == "Figure 1: Example plot."

    paragraph = PaperElement(
        paper_id=paper.id,
        page_number=1,
        element_index=2,
        element_type="paragraph",
        text="This is not a caption.",
    )
    session.add(paragraph)
    session.commit()
    with pytest.raises(VisualAssetError) as error:
        asyncio.run(
            extract_visual_asset(
                session=session,
                storage=storage,
                project_id=project.id,
                paper_id=paper.id,
                page_number=1,
                crop=NormalizedCropBox(0, 0, 1, 1),
                caption_element_id=paragraph.id,
            )
        )
    assert error.value.code == "CAPTION_NOT_FOUND"


@pytest.mark.parametrize(
    "crop",
    [
        NormalizedCropBox(-0.1, 0, 0.5, 1),
        NormalizedCropBox(0, 0, 1.1, 1),
        NormalizedCropBox(0.5, 0, 0.5, 1),
        NormalizedCropBox(0, 0.8, 1, 0.2),
        NormalizedCropBox(0, 0, math.inf, 1),
    ],
)
def test_crop_box_requires_finite_ordered_normalized_coordinates(crop):
    with pytest.raises(VisualAssetError) as error:
        crop.validate()
    assert error.value.code == "INVALID_GEOMETRY"


def test_rejects_oversized_source_and_png(research_db, monkeypatch):
    session, project, paper, _, storage, _ = research_db
    monkeypatch.setattr(visual_assets, "MAX_SOURCE_BYTES", 1)
    with pytest.raises(VisualAssetError) as error:
        asyncio.run(
            extract_visual_asset(
                session=session,
                storage=storage,
                project_id=project.id,
                paper_id=paper.id,
                page_number=1,
                crop=NormalizedCropBox(0, 0, 1, 1),
            )
        )
    assert error.value.code == "SOURCE_TOO_LARGE"

    monkeypatch.setattr(visual_assets, "MAX_SOURCE_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(visual_assets, "MAX_PNG_BYTES", 1)
    with pytest.raises(VisualAssetError) as error:
        asyncio.run(
            extract_visual_asset(
                session=session,
                storage=storage,
                project_id=project.id,
                paper_id=paper.id,
                page_number=1,
                crop=NormalizedCropBox(0, 0, 1, 1),
            )
        )
    assert error.value.code == "OUTPUT_TOO_LARGE"
