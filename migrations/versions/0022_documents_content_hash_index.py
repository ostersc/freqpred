"""documents_content_hash_index

Revision ID: 0022
Revises: 0021
Create Date: 2026-04-04

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0022"
down_revision: Union[str, None] = "0021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_documents_content_hash",
        "documents",
        ["content_hash"],
    )


def downgrade() -> None:
    op.drop_index("ix_documents_content_hash", table_name="documents")
