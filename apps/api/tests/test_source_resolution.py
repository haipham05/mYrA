from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import ChunkElement, Paper, PaperChunk, PaperElement, PaperPage, Project
from app.schemas.evidence import AnchorStatus
from app.services.source_resolution import (
    build_selected_passage_evidence,
    resolve_exact_source_anchor,
)


def _source_rows(db: Session, *, page_text: str = "Alpha evidence. Beta evidence."):
    project = Project(name="Source resolver")
    db.add(project)
    db.flush()
    paper = Paper(
        project_id=project.id,
        filename="source.pdf",
        storage_path="local/source.pdf",
        document_sha256="a" * 64,
        status="READY",
    )
    db.add(paper)
    db.flush()
    page = PaperPage(
        paper_id=paper.id,
        page_number=1,
        width=612,
        height=792,
        raw_text=page_text,
    )
    db.add(page)
    db.add(
        PaperElement(
            paper_id=paper.id,
            page_number=1,
            element_index=0,
            element_type="text",
            text=page_text,
            parser_version="docling-test",
        )
    )
    db.flush()
    return project, paper


def test_resolve_exact_source_anchor_returns_verified_exact_offsets(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'source.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        project, paper = _source_rows(db)
        db.commit()

        anchor = resolve_exact_source_anchor(
            db,
            project_id=project.id,
            paper_id=paper.id,
            page_number=1,
            exact_quote="Beta evidence.",
            document_sha256="A" * 64,
        )

        assert anchor is not None
        assert anchor.anchor_status == AnchorStatus.VERIFIED
        assert anchor.source_char_start == len("Alpha evidence. ")
        assert anchor.source_char_end == len("Alpha evidence. Beta evidence.")
        assert anchor.document_sha256 == paper.document_sha256
        assert anchor.parser_version == "docling-test"
    finally:
        db.close()
        Base.metadata.drop_all(engine)


def test_resolver_rejects_foreign_stale_changed_and_mismatched_sources(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'source-invalid.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        project, paper = _source_rows(db)
        other_project = Project(name="Other project")
        db.add(other_project)
        db.commit()

        args = {
            "paper_id": paper.id,
            "page_number": 1,
            "exact_quote": "Beta evidence.",
        }
        assert resolve_exact_source_anchor(db, project_id=other_project.id, **args) is None
        assert (
            resolve_exact_source_anchor(db, project_id=project.id, document_sha256="b" * 64, **args)
            is None
        )
        assert (
            resolve_exact_source_anchor(
                db,
                project_id=project.id,
                exact_quote="Changed evidence.",
                **{k: v for k, v in args.items() if k != "exact_quote"},
            )
            is None
        )
        assert (
            resolve_exact_source_anchor(
                db,
                project_id=project.id,
                char_start=0,
                char_end=14,
                **args,
            )
            is None
        )
    finally:
        db.close()
        Base.metadata.drop_all(engine)


def test_resolver_rejects_ambiguous_quote_without_exact_offsets(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'source-ambiguous.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        project, paper = _source_rows(db, page_text="Same quote. Other text. Same quote.")
        db.commit()
        assert (
            resolve_exact_source_anchor(
                db,
                project_id=project.id,
                paper_id=paper.id,
                page_number=1,
                exact_quote="Same quote.",
            )
            is None
        )
        start = len("Same quote. Other text. ")
        anchor = resolve_exact_source_anchor(
            db,
            project_id=project.id,
            paper_id=paper.id,
            page_number=1,
            exact_quote="Same quote.",
            char_start=start,
            char_end=start + len("Same quote."),
        )
        assert anchor is not None
        assert anchor.source_char_start == start
    finally:
        db.close()
        Base.metadata.drop_all(engine)


def test_resolver_accepts_only_current_parser_version(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'source-parser.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        project, paper = _source_rows(db)
        db.commit()
        args = {
            "project_id": project.id,
            "paper_id": paper.id,
            "page_number": 1,
            "exact_quote": "Beta evidence.",
        }

        assert resolve_exact_source_anchor(db, parser_version="stale-parser", **args) is None
        anchor = resolve_exact_source_anchor(db, parser_version="docling-test", **args)
        assert anchor is not None
        assert anchor.parser_version == "docling-test"
    finally:
        db.close()
        Base.metadata.drop_all(engine)


def test_resolver_does_not_invent_version_for_legacy_elements(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'source-legacy-parser.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        project, paper = _source_rows(db)
        element = db.query(PaperElement).filter_by(paper_id=paper.id).one()
        element.parser_version = None
        db.commit()

        anchor = resolve_exact_source_anchor(
            db,
            project_id=project.id,
            paper_id=paper.id,
            page_number=1,
            exact_quote="Beta evidence.",
            parser_version="unverifiable-legacy-input",
        )

        assert anchor is not None
        assert anchor.parser_version is None
    finally:
        db.close()
        Base.metadata.drop_all(engine)


def test_selected_passage_evidence_uses_verified_quote_and_linked_chunk(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'selected-evidence.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        project, paper = _source_rows(db)
        paper.title = "A source paper"
        element = db.query(PaperElement).filter_by(paper_id=paper.id).one()
        chunk = PaperChunk(
            paper_id=paper.id,
            chunk_type="child",
            chunk_index=0,
            text=element.text,
        )
        db.add(chunk)
        db.flush()
        db.add(ChunkElement(chunk_id=chunk.id, element_id=element.id, order_index=0))
        db.commit()

        anchor = resolve_exact_source_anchor(
            db,
            project_id=project.id,
            paper_id=paper.id,
            page_number=1,
            exact_quote="Beta evidence.",
            document_sha256=paper.document_sha256,
        )
        assert anchor is not None
        evidence = build_selected_passage_evidence(
            db,
            project_id=project.id,
            paper_id=paper.id,
            anchor=anchor,
        )

        assert evidence is not None
        assert evidence.quote == anchor.exact_quote
        assert evidence.chunk_id == chunk.id
        assert evidence.paper_title == "A source paper"
        assert evidence.anchors == [anchor]
        assert "Alpha evidence. Beta evidence." == evidence.parent_context
    finally:
        db.close()
        Base.metadata.drop_all(engine)
