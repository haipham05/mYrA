"""store provider usage with persisted assistant responses

Revision ID: r7f9b1d3e5a7
Revises: p6e8a0c2d4f6
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "r7f9b1d3e5a7"
down_revision: str | Sequence[str] | None = "p6e8a0c2d4f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("messages") as batch_op:
        batch_op.add_column(sa.Column("provider_usage", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("messages") as batch_op:
        batch_op.drop_column("provider_usage")
