"""track assistant run continuation requests

Revision ID: s8c0e2a4d6f8
Revises: r7f9b1d3e5a7
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "s8c0e2a4d6f8"
down_revision: str | Sequence[str] | None = "r7f9b1d3e5a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "assistant_runs",
        sa.Column("resume_count", sa.Integer(), server_default="0", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("assistant_runs", "resume_count")
