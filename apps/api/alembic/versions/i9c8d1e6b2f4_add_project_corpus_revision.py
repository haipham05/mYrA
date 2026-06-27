"""add project corpus revision for cache invalidation

Revision ID: i9c8d1e6b2f4
Revises: h9b4a1e0d6f2
Create Date: 2026-10-01 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "i9c8d1e6b2f4"
down_revision: str | Sequence[str] | None = "h9b4a1e0d6f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("corpus_revision", sa.BigInteger(), server_default="0", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("projects", "corpus_revision")
