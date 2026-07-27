import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import Paper, PaperElement, PaperPage, Project
from app.services.claim_verification import (
    ClaimAssessment,
    ClaimSource,
    ClaimVerdict,
    verify_claim,
)


def _paper(db, *, text: str, digest: str):
    project = Project(name=f"Project {digest[:5]}")
    db.add(project)
    db.flush()
    paper = Paper(
        project_id=project.id,
        filename="paper.pdf",
        storage_path=f"local/{digest}.pdf",
        document_sha256=digest,
        status="READY",
    )
    db.add(paper)
    db.flush()
    db.add(PaperPage(paper_id=paper.id, page_number=1, width=612, height=792, raw_text=text))
    db.add(
        PaperElement(
            paper_id=paper.id,
            page_number=1,
            element_index=0,
            element_type="text",
            text=text,
            parser_version="test",
        )
    )
    db.flush()
    return project, paper


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'claims.db'}")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)()


def test_supported_requires_current_exact_source_and_checker(tmp_path):
    engine, db = _session(tmp_path)
    try:
        project, paper = _paper(
            db,
            text="The model improves accuracy by 12 percent.",
            digest="a" * 64,
        )
        db.commit()
        result = verify_claim(
            db,
            project_id=project.id,
            claim="The model improves accuracy by 12 percent.",
            selected_paper_ids=[paper.id],
            sources=[
                ClaimSource(
                    evidence_id="E1",
                    paper_id=paper.id,
                    page_number=1,
                    exact_quote="The model improves accuracy by 12 percent.",
                    document_sha256="a" * 64,
                )
            ],
        )
        assert result.verdict is ClaimVerdict.SUPPORTED
        assert len(result.citations) == 1
        assert result.citations[0].anchor_status.value == "verified"
        assert "selected papers" in result.scope_note
    finally:
        db.close()
        Base.metadata.drop_all(engine)


def test_checker_failure_is_not_contradiction_without_explicit_assessment(tmp_path):
    engine, db = _session(tmp_path)
    try:
        project, paper = _paper(
            db,
            text="The model improves accuracy by 12 percent.",
            digest="b" * 64,
        )
        db.commit()
        source = ClaimSource(
            evidence_id="E1",
            paper_id=paper.id,
            page_number=1,
            exact_quote="The model improves accuracy by 12 percent.",
            document_sha256="b" * 64,
        )
        for claim in (
            "The model improves accuracy by 21 percent.",
            "The model does not improve accuracy by 12 percent.",
        ):
            result = verify_claim(
                db,
                project_id=project.id,
                claim=claim,
                selected_paper_ids=[paper.id],
                sources=[source],
            )
            assert result.verdict is ClaimVerdict.INSUFFICIENT
            assert result.citations == []
            assert result.semantic_contradiction is None
    finally:
        db.close()
        Base.metadata.drop_all(engine)


def test_model_assessed_refutation_can_be_contradicted_or_mixed(tmp_path):
    engine, db = _session(tmp_path)
    try:
        project, paper = _paper(
            db,
            text=(
                "The method did not improve accuracy in the trial. "
                "The baseline reached 80 percent accuracy."
            ),
            digest="c" * 64,
        )
        db.commit()
        sources = [
            ClaimSource(
                evidence_id="R1",
                paper_id=paper.id,
                page_number=1,
                exact_quote="The method did not improve accuracy in the trial.",
                document_sha256="c" * 64,
            ),
            ClaimSource(
                evidence_id="S1",
                paper_id=paper.id,
                page_number=1,
                exact_quote="The baseline reached 80 percent accuracy.",
                document_sha256="c" * 64,
            ),
        ]
        claim = "The method improved accuracy in the trial."
        contradicted = verify_claim(
            db,
            project_id=project.id,
            claim=claim,
            selected_paper_ids=[paper.id],
            sources=sources,
            assessment=ClaimAssessment(refuting_evidence_ids=["R1"]),
        )
        assert contradicted.verdict is ClaimVerdict.CONTRADICTED
        assert contradicted.semantic_contradiction == "model_assessed"
        assert [citation.evidence_id for citation in contradicted.citations] == ["R1"]

        mixed = verify_claim(
            db,
            project_id=project.id,
            claim="The baseline reached 80 percent accuracy.",
            selected_paper_ids=[paper.id],
            sources=sources,
            assessment=ClaimAssessment(refuting_evidence_ids=["R1"]),
        )
        assert mixed.verdict is ClaimVerdict.MIXED
        assert {citation.evidence_id for citation in mixed.citations} == {"R1", "S1"}
    finally:
        db.close()
        Base.metadata.drop_all(engine)


def test_stale_hash_foreign_and_unselected_sources_are_not_cited(tmp_path):
    engine, db = _session(tmp_path)
    try:
        project, paper = _paper(
            db,
            text="The method improved accuracy in the trial.",
            digest="d" * 64,
        )
        foreign_project, foreign_paper = _paper(
            db,
            text="The method did not improve accuracy in the trial.",
            digest="e" * 64,
        )
        db.commit()
        sources = [
            ClaimSource(
                evidence_id="STALE",
                paper_id=paper.id,
                page_number=1,
                exact_quote="The method improved accuracy in the trial.",
                document_sha256="f" * 64,
            ),
            ClaimSource(
                evidence_id="FOREIGN_PROJECT",
                paper_id=foreign_paper.id,
                page_number=1,
                exact_quote="The method did not improve accuracy in the trial.",
                document_sha256="e" * 64,
            ),
        ]
        result = verify_claim(
            db,
            project_id=project.id,
            claim="The method improved accuracy in the trial.",
            selected_paper_ids=[paper.id],
            sources=sources,
            assessment=ClaimAssessment(refuting_evidence_ids=["FOREIGN_PROJECT"]),
        )
        assert result.verdict is ClaimVerdict.INSUFFICIENT
        assert result.citations == []

        # Same project but not in the explicit selection is also out of scope.
        other_paper = Paper(
            project_id=project.id,
            filename="other.pdf",
            storage_path="local/other.pdf",
            document_sha256="f" * 64,
            status="READY",
        )
        db.add(other_paper)
        db.flush()
        db.add(
            PaperPage(
                paper_id=other_paper.id,
                page_number=1,
                width=612,
                height=792,
                raw_text="A separate claim is supported.",
            )
        )
        db.commit()
        out_of_selection = verify_claim(
            db,
            project_id=project.id,
            claim="A separate claim is supported.",
            selected_paper_ids=[paper.id],
            sources=[
                ClaimSource(
                    evidence_id="UNSELECTED",
                    paper_id=other_paper.id,
                    page_number=1,
                    exact_quote="A separate claim is supported.",
                    document_sha256="f" * 64,
                )
            ],
        )
        assert out_of_selection.verdict is ClaimVerdict.INSUFFICIENT
        assert out_of_selection.citations == []
    finally:
        db.close()
        Base.metadata.drop_all(engine)


def test_claim_assessment_inputs_are_bounded():
    with pytest.raises(ValidationError):
        ClaimAssessment(refuting_evidence_ids=[f"S{index}" for index in range(19)])
