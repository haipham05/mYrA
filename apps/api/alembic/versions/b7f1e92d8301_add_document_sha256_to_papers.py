"""add_document_sha256_to_papers

Revision ID: b7f1e92d8301
Revises: a6a79bdff6de
Create Date: 2026-09-25 16:35:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7f1e92d8301"
down_revision: str | Sequence[str] | None = "a6a79bdff6de"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: add document_sha256 to papers."""
    op.add_column("papers", sa.Column("document_sha256", sa.String(length=64), nullable=True))


def downgrade() -> None:
    """Downgrade schema: remove document_sha256 from papers."""
    op.drop_column("papers", "document_sha256")
