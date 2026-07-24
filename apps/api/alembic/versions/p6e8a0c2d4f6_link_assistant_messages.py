"""link assistant responses to their durable execution run

Revision ID: p6e8a0c2d4f6
Revises: o5d7f9a1b3c5
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "p6e8a0c2d4f6"
down_revision: str | Sequence[str] | None = "o5d7f9a1b3c5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("messages") as batch_op:
        batch_op.add_column(sa.Column("assistant_run_id", sa.Uuid(), nullable=True))
        batch_op.create_foreign_key(
            "fk_messages_assistant_run_id_assistant_runs",
            "assistant_runs",
            ["assistant_run_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index("ix_messages_assistant_run_id", "messages", ["assistant_run_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_messages_assistant_run_id", table_name="messages")
    with op.batch_alter_table("messages") as batch_op:
        batch_op.drop_constraint("fk_messages_assistant_run_id_assistant_runs", type_="foreignkey")
        batch_op.drop_column("assistant_run_id")
