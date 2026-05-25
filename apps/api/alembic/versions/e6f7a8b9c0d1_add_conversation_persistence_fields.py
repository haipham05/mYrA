"""add conversation persistence fields

Revision ID: e6f7a8b9c0d1
Revises: d4e5f6a7b8c9
Create Date: 2026-09-26 19:10:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e6f7a8b9c0d1"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("is_archived", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.create_index("ix_conversations_is_archived", "conversations", ["is_archived"], unique=False)
    op.add_column("messages", sa.Column("model_name", sa.String(length=100), nullable=True))
    op.add_column("messages", sa.Column("token_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "token_count")
    op.drop_column("messages", "model_name")
    op.drop_index("ix_conversations_is_archived", table_name="conversations")
    op.drop_column("conversations", "is_archived")
