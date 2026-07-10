"""Persist engine checkpoint identities for resumable translation jobs.

Revision ID: k3b5d7f9a1c2
Revises: j2a4c6e8f0b1
Create Date: 2026-10-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "k3b5d7f9a1c2"
down_revision: str | None = "j2a4c6e8f0b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "translation_segments",
        sa.Column("engine_checkpoint_key", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "uq_translation_segment_checkpoint",
        "translation_segments",
        ["translation_id", "engine_checkpoint_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "uq_translation_segment_checkpoint", table_name="translation_segments"
    )
    op.drop_column("translation_segments", "engine_checkpoint_key")
