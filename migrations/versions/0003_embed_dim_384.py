"""Switch documents.embedding from VECTOR(1024) to VECTOR(384).

Replaces Voyage AI (voyage-3, 1024-dim) with local sentence-transformers
(all-MiniLM-L6-v2, 384-dim). The documents table is expected to be empty
at the time of this migration (no data loss).

Also drops and recreates the ivfflat index with the correct dimension.

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-17
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop the old ivfflat index (dimension-specific) if it exists.
    op.execute("DROP INDEX IF EXISTS ix_documents_embedding")

    # Change the column from VECTOR(1024) to VECTOR(384).
    op.alter_column(
        "documents",
        "embedding",
        type_=Vector(384),
        existing_nullable=False,
    )

    # Recreate ivfflat index for the new dimension.
    op.execute(
        "CREATE INDEX ix_documents_embedding ON documents "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_documents_embedding")

    op.alter_column(
        "documents",
        "embedding",
        type_=Vector(1024),
        existing_nullable=False,
    )

    op.execute(
        "CREATE INDEX ix_documents_embedding ON documents "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
