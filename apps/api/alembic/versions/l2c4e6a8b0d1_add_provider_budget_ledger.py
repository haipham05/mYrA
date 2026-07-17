"""add durable provider budget ledger

Revision ID: l2c4e6a8b0d1
Revises: k3b5d7f9a1c2
Create Date: 2026-10-05
"""

import sqlalchemy as sa
from alembic import op

revision = "l2c4e6a8b0d1"
down_revision = "k3b5d7f9a1c2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "provider_budget_days",
        sa.Column("utc_date", sa.Date(), nullable=False),
        sa.Column(
            "committed_usd",
            sa.Numeric(12, 6),
            server_default="0",
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("utc_date"),
    )
    op.create_table(
        "provider_budget_reservations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("utc_date", sa.Date(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("requested_model", sa.String(length=128), nullable=False),
        sa.Column("pricing_snapshot", sa.String(length=64), nullable=False),
        sa.Column("input_bytes", sa.Integer(), nullable=False),
        sa.Column("max_output_tokens", sa.Integer(), nullable=False),
        sa.Column("reserved_usd", sa.Numeric(12, 6), nullable=False),
        sa.Column("settled_estimate_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["utc_date"], ["provider_budget_days.utc_date"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_provider_budget_run_day",
        "provider_budget_reservations",
        ["run_id", "utc_date"],
    )
    op.create_index(
        "ix_provider_budget_day_status",
        "provider_budget_reservations",
        ["utc_date", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_provider_budget_day_status", table_name="provider_budget_reservations")
    op.drop_index("ix_provider_budget_run_day", table_name="provider_budget_reservations")
    op.drop_table("provider_budget_reservations")
    op.drop_table("provider_budget_days")
