"""Add embedding_768 vector(768) column to documents for Ollama/nomic-embed-text.

The existing embedding vector(384) column is left untouched — sentence_transformers
users are unaffected. embedding_768 is nullable; docs are invisible to the 768-dim
retriever until re-embedded via scripts/reindex_embeddings.py.

Revision ID: 0046
Revises: 0045
Create Date: 2026-06-07
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0046"
down_revision = "0045"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("embedding_768", Vector(768), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "embedding_768")
