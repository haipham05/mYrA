"""add nullable trace context to durable queue records

Revision ID: h9b4a1e0d6f2
Revises: g1a2b3c4d5e6
Create Date: 2026-10-01 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "h9b4a1e0d6f2"
down_revision: str | Sequence[str] | None = "g1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for table in ("jobs", "graph_events"):
        op.add_column(table, sa.Column("correlation_id", sa.String(64), nullable=True))
        op.add_column(table, sa.Column("trace_id", sa.String(32), nullable=True))
        op.add_column(table, sa.Column("parent_span_id", sa.String(16), nullable=True))
        op.add_column(
            table,
            sa.Column("trace_sampled", sa.Boolean(), server_default=sa.false(), nullable=False),
        )


def downgrade() -> None:
    for table in ("graph_events", "jobs"):
        op.drop_column(table, "trace_sampled")
        op.drop_column(table, "parent_span_id")
        op.drop_column(table, "trace_id")
        op.drop_column(table, "correlation_id")
