"""add nullable bibliographic metadata to papers

Revision ID: m3a4b5c6d7e8
Revises: l2c4e6a8b0d1
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "m3a4b5c6d7e8"
down_revision: str | Sequence[str] | None = "l2c4e6a8b0d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("title", sa.Text(), nullable=True))
    op.add_column("papers", sa.Column("authors", sa.JSON(), nullable=True))
    op.add_column("papers", sa.Column("publication_year", sa.Integer(), nullable=True))
    op.add_column("papers", sa.Column("doi", sa.String(length=255), nullable=True))
    op.add_column("papers", sa.Column("arxiv_id", sa.String(length=128), nullable=True))
    op.add_column("papers", sa.Column("abstract", sa.Text(), nullable=True))
    op.add_column("papers", sa.Column("source_url", sa.Text(), nullable=True))
    op.add_column("papers", sa.Column("metadata_provenance", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("papers", "metadata_provenance")
    op.drop_column("papers", "source_url")
    op.drop_column("papers", "abstract")
    op.drop_column("papers", "arxiv_id")
    op.drop_column("papers", "doi")
    op.drop_column("papers", "publication_year")
    op.drop_column("papers", "authors")
    op.drop_column("papers", "title")
