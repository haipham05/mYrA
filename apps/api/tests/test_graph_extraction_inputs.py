"""Unit and integration tests for GraphRAG input selector (Task 5.14).

Verifies:
1. Only READY papers selected; PROCESSING or FAILED paper returns empty list.
2. Foreign project rejection: Requesting extraction with Project B's ID for
   Project A's paper fails/returns empty list.
3. Parent-only chunks strictly excluded: A paper having both parent and child chunks
   only selects child chunks.
4. Chunks without linked elements or with empty text are skipped.
5. Bounded batch size: Querying with max_chunks=2 caps returned evidence items to 2
   even if 10 child chunks exist.
6. Evidence items have valid opaque evidence_id (ev_1, ev_2), correct document_sha256,
   and correct page_number.
7. Primary element selection picks the first element by order_index, and parser_version
   defaults to 'v1'.
8. Missing paper or missing document_sha256 returns empty list.
9. Compatibility with the synthetic corpus fixtures.
"""

from __future__ import annotations

from collections.abc import Generator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from tests.fixtures.graphrag.corpus_fixtures import (
    PAPER_A1_ID,
    PROJECT_A_ID,
    PROJECT_B_ID,
    seed_synthetic_corpus,
)

from app.db.base import Base
from app.db.models import ChunkElement, Paper, PaperChunk, PaperElement, PaperPage, Project
from app.services.graphrag import ExtractionEvidenceItem, select_extraction_inputs


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide an isolated, in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def create_test_project(db: Session, name: str = "Test Project") -> Project:
    project = Project(id=uuid4(), name=name)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def create_test_paper(
    db: Session,
    project_id: UUID,
    status: str = "READY",
    document_sha256: str | None = "a" * 64,
    filename: str = "test_paper.pdf",
) -> Paper:
    paper = Paper(
        id=uuid4(),
        project_id=project_id,
        filename=filename,
        storage_path=f"papers/{filename}",
        document_sha256=document_sha256,
        status=status,
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)
    return paper


def create_test_element(
    db: Session,
    paper_id: UUID,
    page_number: int = 1,
    element_index: int = 0,
    text: str = "Sample element text",
    parser_version: str | None = "v1",
) -> PaperElement:
    elem = PaperElement(
        id=uuid4(),
        paper_id=paper_id,
        page_number=page_number,
        element_index=element_index,
        element_type="paragraph",
        text=text,
        parser_version=parser_version,
    )
    db.add(elem)
    db.commit()
    db.refresh(elem)
    return elem


def create_test_chunk(
    db: Session,
    paper_id: UUID,
    chunk_type: str = "child",
    chunk_index: int = 0,
    text: str = "Sample chunk text",
    elements: list[PaperElement] | None = None,
) -> PaperChunk:
    chunk = PaperChunk(
        id=uuid4(),
        paper_id=paper_id,
        chunk_type=chunk_type,
        chunk_index=chunk_index,
        text=text,
    )
    db.add(chunk)
    db.flush()

    if elements:
        for idx, elem in enumerate(elements):
            chunk_elem = ChunkElement(
                id=uuid4(),
                chunk_id=chunk.id,
                element_id=elem.id,
                order_index=idx,
            )
            db.add(chunk_elem)

    db.commit()
    db.refresh(chunk)
    return chunk


def test_only_ready_papers_selected(db_session: Session):
    """Test 1: Only READY papers selected; PROCESSING or FAILED paper returns empty list."""
    project = create_test_project(db_session)

    # 1. PROCESSING paper
    paper_processing = create_test_paper(db_session, project.id, status="PROCESSING")
    elem1 = create_test_element(db_session, paper_processing.id, page_number=1)
    create_test_chunk(db_session, paper_processing.id, elements=[elem1])
    res_processing = select_extraction_inputs(db_session, project.id, paper_processing.id)
    assert res_processing == []

    # 2. FAILED paper
    paper_failed = create_test_paper(db_session, project.id, status="FAILED")
    elem2 = create_test_element(db_session, paper_failed.id, page_number=1)
    create_test_chunk(db_session, paper_failed.id, elements=[elem2])
    res_failed = select_extraction_inputs(db_session, project.id, paper_failed.id)
    assert res_failed == []

    # 3. READY paper
    paper_ready = create_test_paper(db_session, project.id, status="READY")
    elem3 = create_test_element(db_session, paper_ready.id, page_number=1)
    create_test_chunk(db_session, paper_ready.id, elements=[elem3])
    res_ready = select_extraction_inputs(db_session, project.id, paper_ready.id)
    assert len(res_ready) == 1
    assert isinstance(res_ready[0], ExtractionEvidenceItem)


def test_foreign_project_rejection(db_session: Session):
    """Test 2: Foreign project rejection: Requesting extraction with Project B's ID
    for Project A's paper fails/returns empty list.
    """
    project_a = create_test_project(db_session, name="Project A")
    project_b = create_test_project(db_session, name="Project B")

    paper_a = create_test_paper(db_session, project_a.id, status="READY")
    elem = create_test_element(db_session, paper_a.id, page_number=1)
    create_test_chunk(db_session, paper_a.id, elements=[elem])

    # Request extraction with Project B's ID for Project A's paper
    result = select_extraction_inputs(db_session, project_b.id, paper_a.id)
    assert result == []


def test_parent_only_chunks_strictly_excluded(db_session: Session):
    """Test 3: Parent-only chunks strictly excluded: A paper having both parent and child
    chunks only selects child chunks (chunk_type == 'child').
    """
    project = create_test_project(db_session)
    paper = create_test_paper(db_session, project.id, status="READY")

    elem1 = create_test_element(db_session, paper.id, page_number=1, element_index=0)
    elem2 = create_test_element(db_session, paper.id, page_number=1, element_index=1)

    # Parent chunk
    parent_chunk = create_test_chunk(
        db_session,
        paper.id,
        chunk_type="parent",
        chunk_index=0,
        text="Parent chunk text containing overarching context.",
        elements=[elem1, elem2],
    )
    # Child chunks
    child_chunk1 = create_test_chunk(
        db_session,
        paper.id,
        chunk_type="child",
        chunk_index=1,
        text="Child chunk 1 text.",
        elements=[elem1],
    )
    child_chunk2 = create_test_chunk(
        db_session,
        paper.id,
        chunk_type="child",
        chunk_index=2,
        text="Child chunk 2 text.",
        elements=[elem2],
    )

    items = select_extraction_inputs(db_session, project.id, paper.id)
    assert len(items) == 2
    selected_chunk_ids = {item.chunk_id for item in items}
    assert parent_chunk.id not in selected_chunk_ids
    assert child_chunk1.id in selected_chunk_ids
    assert child_chunk2.id in selected_chunk_ids


def test_chunks_without_linked_elements_or_empty_text_are_skipped(db_session: Session):
    """Test 4: Chunks without linked elements or with empty text are skipped."""
    project = create_test_project(db_session)
    paper = create_test_paper(db_session, project.id, status="READY")

    elem1 = create_test_element(db_session, paper.id, page_number=1, element_index=0)
    elem2 = create_test_element(db_session, paper.id, page_number=1, element_index=1)

    # 1. Valid chunk
    c1 = create_test_chunk(
        db_session,
        paper.id,
        chunk_type="child",
        chunk_index=0,
        text="Valid chunk text",
        elements=[elem1],
    )
    # 2. Chunk without any linked ChunkElement
    c2 = create_test_chunk(
        db_session,
        paper.id,
        chunk_type="child",
        chunk_index=1,
        text="Chunk without linked element",
        elements=[],
    )
    # 3. Chunk with empty text
    create_test_chunk(
        db_session,
        paper.id,
        chunk_type="child",
        chunk_index=2,
        text="",
        elements=[elem2],
    )
    # 4. Chunk with whitespace-only text
    create_test_chunk(
        db_session,
        paper.id,
        chunk_type="child",
        chunk_index=3,
        text="   \n\t  ",
        elements=[elem2],
    )
    # 5. Another valid chunk
    c5 = create_test_chunk(
        db_session,
        paper.id,
        chunk_type="child",
        chunk_index=4,
        text="Another valid chunk text",
        elements=[elem2],
    )

    items = select_extraction_inputs(db_session, project.id, paper.id)
    assert len(items) == 2
    assert items[0].chunk_id == c1.id
    assert items[0].evidence_id == "ev_1"
    assert items[1].chunk_id == c5.id
    assert items[1].evidence_id == "ev_2"
    assert c2.id not in {item.chunk_id for item in items}


def test_bounded_batch_size_caps_items(db_session: Session):
    """Test 5: Bounded batch size: Querying with max_chunks=2 caps the returned
    evidence items to 2 even if 10 child chunks exist.
    """
    project = create_test_project(db_session)
    paper = create_test_paper(db_session, project.id, status="READY")

    chunk_ids = []
    for i in range(10):
        elem = create_test_element(
            db_session, paper.id, page_number=1, element_index=i, text=f"Element text {i}"
        )
        chunk = create_test_chunk(
            db_session,
            paper.id,
            chunk_type="child",
            chunk_index=i,
            text=f"Child chunk text {i}",
            elements=[elem],
        )
        chunk_ids.append(chunk.id)

    items = select_extraction_inputs(db_session, project.id, paper.id, max_chunks=2)
    assert len(items) == 2
    assert items[0].chunk_id == chunk_ids[0]
    assert items[1].chunk_id == chunk_ids[1]
    assert items[0].chunk_index == 0
    assert items[1].chunk_index == 1
    assert items[0].evidence_id == "ev_1"
    assert items[1].evidence_id == "ev_2"


def test_evidence_items_opaque_ids_and_correct_metadata(db_session: Session):
    """Test 6: Evidence items have valid opaque evidence_id (ev_1, ev_2),
    correct document_sha256, and correct page_number.
    """
    project = create_test_project(db_session)
    doc_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    paper = create_test_paper(db_session, project.id, status="READY", document_sha256=doc_sha)

    elem_page2 = create_test_element(
        db_session,
        paper.id,
        page_number=2,
        element_index=0,
        text="Element on page 2",
        parser_version="v2.1",
    )
    elem_page3 = create_test_element(
        db_session,
        paper.id,
        page_number=3,
        element_index=1,
        text="Element on page 3",
        parser_version="v2.1",
    )

    chunk1 = create_test_chunk(
        db_session,
        paper.id,
        chunk_type="child",
        chunk_index=0,
        text="Chunk covering page 2 content",
        elements=[elem_page2],
    )
    chunk2 = create_test_chunk(
        db_session,
        paper.id,
        chunk_type="child",
        chunk_index=1,
        text="Chunk covering page 3 content",
        elements=[elem_page3],
    )

    items = select_extraction_inputs(db_session, project.id, paper.id)
    assert len(items) == 2

    item1 = items[0]
    assert item1.evidence_id == "ev_1"
    assert item1.chunk_id == chunk1.id
    assert item1.text == "Chunk covering page 2 content"
    assert item1.page_number == 2
    assert item1.element_id == elem_page2.id
    assert item1.parser_version == "v2.1"
    assert item1.document_sha256 == doc_sha
    assert item1.chunk_index == 0

    item2 = items[1]
    assert item2.evidence_id == "ev_2"
    assert item2.chunk_id == chunk2.id
    assert item2.text == "Chunk covering page 3 content"
    assert item2.page_number == 3
    assert item2.element_id == elem_page3.id
    assert item2.parser_version == "v2.1"
    assert item2.document_sha256 == doc_sha
    assert item2.chunk_index == 1


def test_evidence_item_carries_all_linked_page_and_element_sources(db_session: Session):
    project = create_test_project(db_session)
    paper = create_test_paper(db_session, project.id)
    first = create_test_element(
        db_session, paper.id, page_number=1, element_index=0, text="First page element"
    )
    second = create_test_element(
        db_session, paper.id, page_number=2, element_index=1, text="Second page element"
    )
    db_session.add_all(
        [
            PaperPage(
                id=uuid4(),
                paper_id=paper.id,
                page_number=1,
                width=612,
                height=792,
                raw_text="First page text",
            ),
            PaperPage(
                id=uuid4(),
                paper_id=paper.id,
                page_number=2,
                width=612,
                height=792,
                raw_text="Second page text",
            ),
        ]
    )
    db_session.commit()
    create_test_chunk(
        db_session,
        paper.id,
        chunk_type="child",
        chunk_index=0,
        text="A chunk spanning two pages",
        elements=[first, second],
    )

    [item] = select_extraction_inputs(db_session, project.id, paper.id)

    assert [(page.page_number, page.raw_text) for page in item.source_pages] == [
        (1, "First page text"),
        (2, "Second page text"),
    ]
    assert [element.element_id for element in item.source_elements] == [first.id, second.id]


def test_primary_element_ordered_selection_and_default_parser_version(db_session: Session):
    """Test that the primary element is the one with lowest order_index,
    and parser_version defaults to 'v1'.
    """
    project = create_test_project(db_session)
    paper = create_test_paper(db_session, project.id, status="READY")

    elem_first = create_test_element(
        db_session,
        paper.id,
        page_number=4,
        element_index=10,
        text="Primary element",
        parser_version=None,
    )
    elem_second = create_test_element(
        db_session,
        paper.id,
        page_number=5,
        element_index=11,
        text="Secondary element",
        parser_version="v99",
    )

    # Attach in order_index 0 and 1
    create_test_chunk(
        db_session,
        paper.id,
        chunk_type="child",
        chunk_index=0,
        text="Multi-element chunk text",
        elements=[elem_first, elem_second],
    )

    items = select_extraction_inputs(db_session, project.id, paper.id)
    assert len(items) == 1
    assert items[0].element_id == elem_first.id
    assert items[0].page_number == 4
    assert items[0].parser_version == "v1"


def test_missing_paper_or_missing_sha256_returns_empty(db_session: Session):
    """Test that non-existent paper, or paper with missing document_sha256, returns empty list."""
    project = create_test_project(db_session)

    # 1. Non-existent paper
    assert select_extraction_inputs(db_session, project.id, uuid4()) == []

    # 2. Paper without document_sha256
    paper_no_sha = create_test_paper(db_session, project.id, status="READY", document_sha256=None)
    elem = create_test_element(db_session, paper_no_sha.id)
    create_test_chunk(db_session, paper_no_sha.id, elements=[elem])
    assert select_extraction_inputs(db_session, project.id, paper_no_sha.id) == []

    # 3. max_chunks <= 0
    paper_ready = create_test_paper(db_session, project.id, status="READY")
    elem2 = create_test_element(db_session, paper_ready.id)
    create_test_chunk(db_session, paper_ready.id, elements=[elem2])
    assert select_extraction_inputs(db_session, project.id, paper_ready.id, max_chunks=0) == []
    assert select_extraction_inputs(db_session, project.id, paper_ready.id, max_chunks=-5) == []

    # 4. String UUIDs accepted
    str_items = select_extraction_inputs(
        db_session,
        str(project.id),
        str(paper_ready.id),  # type: ignore[arg-type]
    )
    assert len(str_items) == 1


def test_synthetic_corpus_integration():
    """Verify select_extraction_inputs against synthetic corpus fixtures."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        seed_synthetic_corpus(session)

        # Select inputs for Paper A1 in Project A
        items = select_extraction_inputs(session, PROJECT_A_ID, PAPER_A1_ID, max_chunks=10)
        assert len(items) > 0

        for idx, item in enumerate(items):
            assert item.evidence_id == f"ev_{idx + 1}"
            assert len(item.document_sha256) == 64
            assert item.page_number >= 1
            assert len(item.text.strip()) > 0
            assert item.element_id is not None

        # Cross project rejection on synthetic corpus
        cross_res = select_extraction_inputs(session, PROJECT_B_ID, PAPER_A1_ID)
        assert cross_res == []
    finally:
        session.close()
        engine.dispose()
