"""Add tv_query column to catalyst_queries.

The catalyst generator now emits dual-format queries: query_text for
web-search fetchers (Tavily, NewsAPI, GDELT, Reddit) and tv_query for
the Internet Archive TV News Archive fetcher (Solr/Lucene boolean syntax).
tv_query is nullable — it will be None for markets where TV transcripts
are not a relevant signal source.

Revision ID: 0011
Revises: 0010
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("catalyst_queries", sa.Column("tv_query", sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column("catalyst_queries", "tv_query")
