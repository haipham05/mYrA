"""add_raw_text_to_paper_pages

Revision ID: d4e5f6a7b8c9
Revises: c1a2e34d5678
Create Date: 2026-09-26 14:10:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: str | Sequence[str] | None = "c1a2e34d5678"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: add raw_text to paper_pages."""
    op.add_column("paper_pages", sa.Column("raw_text", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema: remove raw_text from paper_pages."""
    op.drop_column("paper_pages", "raw_text")
