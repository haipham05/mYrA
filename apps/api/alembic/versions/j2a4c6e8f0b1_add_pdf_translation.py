"""add durable PDF translation jobs, checkpoints and glossary

Revision ID: j2a4c6e8f0b1
Revises: i9c8d1e6b2f4
Create Date: 2026-10-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "j2a4c6e8f0b1"
down_revision: str | None = "i9c8d1e6b2f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "translation_documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("stage", sa.String(length=40), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("acknowledge_external_processing", sa.Boolean(), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_storage_path", sa.String(length=1000), nullable=False),
        sa.Column("source_filename", sa.String(length=500), nullable=False),
        sa.Column("source_page_count", sa.Integer(), nullable=True),
        sa.Column("source_map", sa.JSON(), nullable=False),
        sa.Column("glossary_snapshot", sa.JSON(), nullable=False),
        sa.Column("engine_version", sa.String(length=50), nullable=False),
        sa.Column("babeldoc_version", sa.String(length=50), nullable=False),
        sa.Column("provider_policy_version", sa.String(length=50), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("attempt_token", sa.String(length=64), nullable=True),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_units", sa.Integer(), nullable=False),
        sa.Column("total_units", sa.Integer(), nullable=True),
        sa.Column("output_storage_path", sa.String(length=1000), nullable=True),
        sa.Column("output_sha256", sa.String(length=64), nullable=True),
        sa.Column("error_code", sa.String(length=80), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("is_retryable", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id", "paper_id", "idempotency_key", name="uq_translation_request_key"
        ),
    )
    op.create_index("ix_translation_documents_project_id", "translation_documents", ["project_id"])
    op.create_index("ix_translation_documents_paper_id", "translation_documents", ["paper_id"])
    op.create_index("ix_translation_documents_status", "translation_documents", ["status"])
    op.create_index(
        "ix_translation_queue",
        "translation_documents",
        ["status", "lease_expires_at", "created_at"],
    )
    op.create_table(
        "translation_segments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("translation_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("source_page_number", sa.Integer(), nullable=False),
        sa.Column("source_element_id", sa.Uuid(), nullable=True),
        sa.Column("source_text_hash", sa.String(length=64), nullable=False),
        sa.Column("source_quote", sa.Text(), nullable=False),
        sa.Column("translated_text", sa.Text(), nullable=False),
        sa.Column("translated_text_hash", sa.String(length=64), nullable=False),
        sa.Column("output_page_number", sa.Integer(), nullable=True),
        sa.Column("output_boxes", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["translation_id"], ["translation_documents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("translation_id", "ordinal", name="uq_translation_segment_ordinal"),
    )
    op.create_index(
        "ix_translation_segments_translation_id", "translation_segments", ["translation_id"]
    )
    op.create_table(
        "project_translation_glossary_entries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("source_term", sa.String(length=255), nullable=False),
        sa.Column("preferred_translation", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "source_term", name="uq_project_glossary_source_term"),
    )
    op.create_index(
        "ix_project_translation_glossary_entries_project_id",
        "project_translation_glossary_entries",
        ["project_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_project_translation_glossary_entries_project_id",
        table_name="project_translation_glossary_entries",
    )
    op.drop_table("project_translation_glossary_entries")
    op.drop_index("ix_translation_segments_translation_id", table_name="translation_segments")
    op.drop_table("translation_segments")
    op.drop_index("ix_translation_queue", table_name="translation_documents")
    op.drop_index("ix_translation_documents_status", table_name="translation_documents")
    op.drop_index("ix_translation_documents_paper_id", table_name="translation_documents")
    op.drop_index("ix_translation_documents_project_id", table_name="translation_documents")
    op.drop_table("translation_documents")
