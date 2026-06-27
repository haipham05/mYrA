from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.crud.corpus import bump_corpus_revision, read_corpus_revision
from app.crud.paper import create_paper, update_paper_status
from app.db.base import Base
from app.db.models import Project


def test_corpus_revision_is_atomic_transactional_and_starts_at_zero() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            project = Project(name="revision")
            db.add(project)
            db.commit()
            project_id = project.id
            assert read_corpus_revision(db, project_id) == 0

            bump_corpus_revision(db, project_id)
            db.commit()
            assert read_corpus_revision(db, project_id) == 1

            bump_corpus_revision(db, project_id)
            db.rollback()
            assert read_corpus_revision(db, project_id) == 1

        # Separate sessions execute SQL-side increments, not read/modify/write values.
        with Session(engine) as first:
            bump_corpus_revision(first, project_id)
            first.commit()
        with Session(engine) as second:
            bump_corpus_revision(second, project_id)
            second.commit()
        with Session(engine) as verify:
            assert read_corpus_revision(verify, project_id) == 3

        with Session(engine) as db:
            paper = create_paper(db, project_id, "paper.pdf", "papers/paper.pdf")
            assert read_corpus_revision(db, project_id) == 3
            update_paper_status(db, paper.id, "READY")
            assert read_corpus_revision(db, project_id) == 4
            update_paper_status(db, paper.id, "READY")
            assert read_corpus_revision(db, project_id) == 4
            update_paper_status(db, paper.id, "FAILED")
            assert read_corpus_revision(db, project_id) == 5
            create_paper(db, project_id, "ready.pdf", "papers/ready.pdf", status="READY")
            assert read_corpus_revision(db, project_id) == 6
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()
