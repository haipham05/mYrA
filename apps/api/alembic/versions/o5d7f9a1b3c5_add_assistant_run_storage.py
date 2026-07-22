"""add durable assistant run, step, and approval storage

Revision ID: o5d7f9a1b3c5
Revises: n4b6c8d0e2f3
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "o5d7f9a1b3c5"
down_revision: str | Sequence[str] | None = "n4b6c8d0e2f3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "assistant_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("parent_run_id", sa.Uuid(), nullable=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("request_payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="QUEUED", nullable=False),
        sa.Column("intent", sa.String(length=40), nullable=True),
        sa.Column("route_decision", sa.JSON(), nullable=True),
        sa.Column("action_summary", sa.String(length=240), nullable=True),
        sa.Column("current_stage", sa.String(length=80), nullable=True),
        sa.Column("result_payload", sa.JSON(), nullable=True),
        sa.Column("safe_error", sa.String(length=120), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("lease_owner", sa.String(length=100), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["parent_run_id"], ["assistant_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id", "idempotency_key", name="uq_assistant_runs_conversation_idempotency"
        ),
    )
    op.create_index("ix_assistant_runs_status", "assistant_runs", ["status"])
    op.create_index("ix_assistant_runs_status_created", "assistant_runs", ["status", "created_at"])
    op.create_index(
        "ix_assistant_runs_project_created", "assistant_runs", ["project_id", "created_at"]
    )

    op.create_table(
        "assistant_run_steps",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("step_key", sa.String(length=100), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("tool_name", sa.String(length=80), nullable=True),
        sa.Column("status", sa.String(length=24), server_default="PENDING", nullable=False),
        sa.Column("input_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("output_payload", sa.JSON(), nullable=True),
        sa.Column("external_effect_id", sa.String(length=255), nullable=True),
        sa.Column("safe_error", sa.String(length=120), nullable=True),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["assistant_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "step_key", name="uq_assistant_run_steps_run_key"),
    )
    op.create_index("ix_assistant_run_steps_status", "assistant_run_steps", ["status"])

    op.create_table(
        "assistant_approval_actions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("action_type", sa.String(length=80), nullable=False),
        sa.Column("arguments", sa.JSON(), nullable=False),
        sa.Column("source_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=24), server_default="PENDING", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["assistant_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id", "idempotency_key", name="uq_assistant_actions_run_idempotency"
        ),
    )
    op.create_index(
        "ix_assistant_actions_status_expires",
        "assistant_approval_actions",
        ["status", "expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_assistant_actions_status_expires", table_name="assistant_approval_actions")
    op.drop_table("assistant_approval_actions")
    op.drop_index("ix_assistant_run_steps_status", table_name="assistant_run_steps")
    op.drop_table("assistant_run_steps")
    op.drop_index("ix_assistant_runs_project_created", table_name="assistant_runs")
    op.drop_index("ix_assistant_runs_status_created", table_name="assistant_runs")
    op.drop_index("ix_assistant_runs_status", table_name="assistant_runs")
    op.drop_table("assistant_runs")
