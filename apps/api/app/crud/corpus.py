"""Atomic project corpus revision helpers used by cache-key construction."""

from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import Project


def read_corpus_revision(db: Session, project_id: UUID) -> int:
    """Read the current revision from the database, bypassing stale ORM identity state."""
    revision = db.scalar(
        select(Project.corpus_revision)
        .where(Project.id == project_id)
        .execution_options(populate_existing=True)
    )
    if revision is None:
        raise ValueError("Project does not exist")
    return revision


def bump_corpus_revision(db: Session, project_id: UUID) -> None:
    """Increment atomically in the caller's transaction; never commit independently."""
    result = db.execute(
        update(Project)
        .where(Project.id == project_id)
        .values(corpus_revision=Project.corpus_revision + 1)
    )
    if result.rowcount != 1:
        raise ValueError("Project does not exist")
