"""mark_kalshi_changelog_reviewed_2026_06_23

Revision ID: 0049
Revises: 0048
Create Date: 2026-06-23 22:12:50.687684

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0049'
down_revision: str | None = '0048'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET last_reviewed_at = '2026-06-23',"
        "     unreviewed_count = 0,"
        "     last_checked_at = NULL"
        " WHERE id = 1"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET last_reviewed_at = '2026-06-22',"
        "     unreviewed_count = 2,"
        "     last_checked_at = NULL"
        " WHERE id = 1"
    )
