import csv
import io

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import Paper, Project, ResearchArtifactRevision
from app.services.artifact_export import export_bibtex, export_comparison_csv, export_markdown


def test_markdown_export_preserves_citation_and_source_reference() -> None:
    revision = ResearchArtifactRevision(
        title="Research findings",
        payload={
            "report_markdown": "The result is supported [E1]. Local file /home/user/private.pdf "
            "and sk-123456789012345678901234 must not leak."
        },
        source_manifest=[
            {
                "evidence_id": "E1",
                "paper_title": "A Research Paper",
                "page_number": 4,
                "quote": "The exact source text.",
            }
        ],
    )

    exported = export_markdown(revision, "report")

    assert "The result is supported [E1]." in exported
    assert "[E1] A Research Paper, p. 4" in exported
    assert "The exact source text." in exported
    assert "/home/user/private.pdf" not in exported
    assert "sk-123456789012345678901234" not in exported


def test_comparison_csv_keeps_each_cell_source_identifier() -> None:
    revision = ResearchArtifactRevision(
        title="Comparison",
        payload={
            "matrix": {
                "cells": [
                    {
                        "paper_id": "paper-1",
                        "dimension": "method",
                        "status": "evidence_available",
                        "excerpts": [
                            {
                                "evidence": {
                                    "id": "C1",
                                    "paper_id": "paper-1",
                                    "page_number": 5,
                                    "document_sha256": "a" * 64,
                                    "quote": "A sourced comparison cell.",
                                }
                            }
                        ],
                    }
                ]
            }
        },
    )

    exported = export_comparison_csv(revision)

    assert "paper_id,dimension,status,evidence_id,page,document_sha256,quote" in exported
    assert "paper-1,method,evidence_available,C1,5," in exported
    assert "A sourced comparison cell." in exported


def test_comparison_csv_neutralizes_formula_cells_even_after_controls() -> None:
    revision = ResearchArtifactRevision(
        title="Comparison",
        payload={
            "matrix": {
                "cells": [
                    {
                        "paper_id": "paper-1",
                        "dimension": "method",
                        "status": "evidence_available",
                        "excerpts": [
                            {
                                "evidence": {
                                    "id": "C1",
                                    "paper_id": "paper-1",
                                    "page_number": 5,
                                    "quote": ' \x01=HYPERLINK("https://bad.example")',
                                }
                            }
                        ],
                    }
                ]
            }
        },
    )

    rows = list(csv.reader(io.StringIO(export_comparison_csv(revision))))

    assert rows[1][0:4] == ["paper-1", "method", "evidence_available", "C1"]
    assert rows[1][6].startswith("' \x01=HYPERLINK")


def test_bibtex_export_omits_unknown_bibliographic_fields(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'bibtex.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        project = Project(name="Bibliography")
        db.add(project)
        db.flush()
        paper = Paper(
            project_id=project.id,
            filename="paper.pdf",
            storage_path="paper.pdf",
            title="Known Paper Title",
            authors=["A. Author", "B. Author"],
            publication_year=None,
            doi=None,
        )
        db.add(paper)
        db.commit()
        revision = ResearchArtifactRevision(
            title="Report",
            payload={},
            source_manifest=[{"paper_id": str(paper.id)}],
        )

        exported = export_bibtex(db, revision, project.id)

        assert "title = {Known Paper Title}" in exported
        assert "author = {A. Author and B. Author}" in exported
        assert "year =" not in exported
        assert "doi =" not in exported
    finally:
        db.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
