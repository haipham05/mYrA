"""add versioned research artifacts

Revision ID: t9a1c3e5f7b2
Revises: s8c0e2a4d6f8
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "t9a1c3e5f7b2"
down_revision: str | Sequence[str] | None = "s8c0e2a4d6f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("artifact_type", sa.String(length=40), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("latest_revision", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_research_artifacts_project_id", "research_artifacts", ["project_id"])
    op.create_index(
        "ix_research_artifacts_project_updated",
        "research_artifacts",
        ["project_id", "updated_at"],
    )
    op.create_table(
        "research_artifact_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("artifact_id", sa.Uuid(), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("scope_snapshot", sa.JSON(), nullable=False),
        sa.Column("source_manifest", sa.JSON(), nullable=False),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("usage", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["research_artifacts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("artifact_id", "revision_number", name="uq_artifact_revision_number"),
    )
    op.create_index(
        "ix_research_artifact_revisions_artifact_id",
        "research_artifact_revisions",
        ["artifact_id"],
    )
    op.create_index(
        "ix_artifact_revisions_artifact_created",
        "research_artifact_revisions",
        ["artifact_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_artifact_revisions_artifact_created", table_name="research_artifact_revisions"
    )
    op.drop_index(
        "ix_research_artifact_revisions_artifact_id", table_name="research_artifact_revisions"
    )
    op.drop_table("research_artifact_revisions")
    op.drop_index("ix_research_artifacts_project_updated", table_name="research_artifacts")
    op.drop_index("ix_research_artifacts_project_id", table_name="research_artifacts")
    op.drop_table("research_artifacts")
