"""add graph events and snapshots

Revision ID: g1a2b3c4d5e6
Revises: f1a2b3c4d5e6
Create Date: 2026-09-29 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "g1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create graph_events table
    op.create_table(
        "graph_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("action", sa.String(length=50), server_default="UPSERT", nullable=False),
        sa.Column("generation_id", sa.String(length=100), nullable=False),
        sa.Column("ontology_version", sa.String(length=50), server_default="1.0.0", nullable=False),
        sa.Column("extractor_version", sa.String(length=50), server_default="1.0.0", nullable=False),
        sa.Column("status", sa.String(length=50), server_default="PENDING", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default="3", nullable=False),
        sa.Column("lease_owner", sa.String(length=255), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("paper_id", "generation_id", "action", name="uq_graph_events_paper_generation_action"),
    )
    op.create_index("ix_graph_events_project_id", "graph_events", ["project_id"], unique=False)
    op.create_index("ix_graph_events_paper_id", "graph_events", ["paper_id"], unique=False)
    op.create_index("ix_graph_events_generation_id", "graph_events", ["generation_id"], unique=False)
    op.create_index("ix_graph_events_status", "graph_events", ["status"], unique=False)
    op.create_index("ix_graph_events_lease_expires_at", "graph_events", ["lease_expires_at"], unique=False)

    # 2. Create graph_fact_snapshots table
    op.create_table(
        "graph_fact_snapshots",
        sa.Column("fact_id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("generation_id", sa.String(length=100), nullable=False),
        sa.Column("event_id", sa.Uuid(), nullable=True),
        sa.Column("subject_key", sa.String(length=128), nullable=False),
        sa.Column("subject_name", sa.String(length=255), nullable=False),
        sa.Column("subject_type", sa.String(length=50), nullable=False),
        sa.Column("predicate", sa.String(length=50), nullable=False),
        sa.Column("object_key", sa.String(length=128), nullable=False),
        sa.Column("object_name", sa.String(length=255), nullable=False),
        sa.Column("object_type", sa.String(length=50), nullable=False),
        sa.Column("qualifiers", sa.JSON(), nullable=True),
        sa.Column("chunk_id", sa.Uuid(), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("element_id", sa.Uuid(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("exact_quote", sa.Text(), nullable=False),
        sa.Column("document_sha256", sa.String(length=64), nullable=False),
        sa.Column("validation_version", sa.String(length=50), server_default="1.0.0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["event_id"], ["graph_events.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("fact_id"),
    )
    op.create_index("ix_graph_fact_snapshots_project_id", "graph_fact_snapshots", ["project_id"], unique=False)
    op.create_index("ix_graph_fact_snapshots_paper_id", "graph_fact_snapshots", ["paper_id"], unique=False)
    op.create_index("ix_graph_fact_snapshots_generation_id", "graph_fact_snapshots", ["generation_id"], unique=False)
    op.create_index("ix_graph_fact_snapshots_event_id", "graph_fact_snapshots", ["event_id"], unique=False)
    op.create_index("ix_graph_fact_snapshots_subject_key", "graph_fact_snapshots", ["subject_key"], unique=False)
    op.create_index("ix_graph_fact_snapshots_predicate", "graph_fact_snapshots", ["predicate"], unique=False)
    op.create_index("ix_graph_fact_snapshots_object_key", "graph_fact_snapshots", ["object_key"], unique=False)


def downgrade() -> None:
    op.drop_table("graph_fact_snapshots")
    op.drop_table("graph_events")
