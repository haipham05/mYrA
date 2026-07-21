"""persist paper scope on conversations

Revision ID: n4b6c8d0e2f3
Revises: m3a4b5c6d7e8
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "n4b6c8d0e2f3"
down_revision: str | Sequence[str] | None = "m3a4b5c6d7e8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("paper_scope", sa.String(length=20), server_default="project", nullable=False),
    )
    op.add_column("conversations", sa.Column("selected_paper_ids", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("conversations", "selected_paper_ids")
    op.drop_column("conversations", "paper_scope")
