"""add_vector_and_fts_to_chunks

Revision ID: c1a2e34d5678
Revises: b7f1e92d8301
Create Date: 2026-09-25 16:50:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c1a2e34d5678"
down_revision: str | Sequence[str] | None = "b7f1e92d8301"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: add halfvec(1024) embedding and full-text search tsvector to paper_chunks."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        op.execute("ALTER TABLE paper_chunks ADD COLUMN IF NOT EXISTS embedding_vec halfvec(1024);")
        op.execute(
            "ALTER TABLE paper_chunks ADD COLUMN IF NOT EXISTS tsv_content tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED;"
        )
        op.execute("CREATE INDEX IF NOT EXISTS ix_paper_chunks_tsv ON paper_chunks USING gin(tsv_content);")
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_paper_chunks_embedding_vec ON paper_chunks "
            "USING hnsw (embedding_vec halfvec_cosine_ops);"
        )
        # Backfill existing JSON embeddings into native halfvec(1024)
        op.execute(
            "UPDATE paper_chunks "
            "SET embedding_vec = ('[' || array_to_string(ARRAY(SELECT json_array_elements_text(embedding)), ',') || ']')::halfvec "
            "WHERE embedding IS NOT NULL AND embedding_vec IS NULL;"
        )
    else:
        op.add_column("paper_chunks", sa.Column("embedding_vec", sa.Text(), nullable=True))
        op.add_column("paper_chunks", sa.Column("tsv_content", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema: remove vector and tsvector from paper_chunks."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_paper_chunks_embedding_vec;")
        op.execute("DROP INDEX IF EXISTS ix_paper_chunks_tsv;")
        op.execute("ALTER TABLE paper_chunks DROP COLUMN IF EXISTS tsv_content;")
        op.execute("ALTER TABLE paper_chunks DROP COLUMN IF EXISTS embedding_vec;")
    else:
        op.drop_column("paper_chunks", "tsv_content")
        op.drop_column("paper_chunks", "embedding_vec")
