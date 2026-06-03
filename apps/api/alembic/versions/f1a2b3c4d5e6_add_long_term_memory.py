"""add long term memory tables

Revision ID: f1a2b3c4d5e6
Revises: a1b2c3d4e5f6
Create Date: 2026-09-27 11:00:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create memories table
    op.create_table(
        "memories",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("memory_type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=50), server_default="ACTIVE", nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("importance", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("is_pinned", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("superseded_by_id", sa.Uuid(), nullable=True),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["superseded_by_id"], ["memories.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memories_project_id", "memories", ["project_id"], unique=False)
    op.create_index("ix_memories_memory_type", "memories", ["memory_type"], unique=False)
    op.create_index("ix_memories_status", "memories", ["status"], unique=False)
    op.create_index("ix_memories_is_pinned", "memories", ["is_pinned"], unique=False)
    op.create_index("ix_memories_project_status", "memories", ["project_id", "status"], unique=False)
    op.create_index("ix_memories_project_type", "memories", ["project_id", "memory_type"], unique=False)

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding_vec halfvec(1024);")
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_memories_embedding_vec ON memories "
            "USING hnsw (embedding_vec halfvec_cosine_ops);"
        )
    else:
        op.add_column("memories", sa.Column("embedding_vec", sa.Text(), nullable=True))

    # 2. Create memory_sources table
    op.create_table(
        "memory_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("memory_id", sa.Uuid(), nullable=False),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=True),
        sa.Column("paper_id", sa.Uuid(), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("quote_text", sa.Text(), nullable=True),
        sa.Column("document_sha256", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["memory_id"], ["memories.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_sources_memory_id", "memory_sources", ["memory_id"], unique=False)
    op.create_index("ix_memory_sources_message_id", "memory_sources", ["message_id"], unique=False)
    op.create_index("ix_memory_sources_paper_id", "memory_sources", ["paper_id"], unique=False)

    # 3. Create memory_audits table
    op.create_table(
        "memory_audits",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("memory_id", sa.Uuid(), nullable=False),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("old_content", sa.Text(), nullable=True),
        sa.Column("new_content", sa.Text(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["memory_id"], ["memories.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_audits_memory_id", "memory_audits", ["memory_id"], unique=False)


def downgrade() -> None:
    op.drop_table("memory_audits")
    op.drop_table("memory_sources")
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_memories_embedding_vec;")
        op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS embedding_vec;")
    op.drop_table("memories")
