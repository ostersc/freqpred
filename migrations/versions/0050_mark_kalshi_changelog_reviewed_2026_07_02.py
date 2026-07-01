"""mark_kalshi_changelog_reviewed_2026_07_02

Revision ID: 0050
Revises: 0049
Create Date: 2026-06-24 00:00:00.000000

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0050'
down_revision: str | None = '0049'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET last_reviewed_at = '2026-07-02',"
        "     unreviewed_count = 0,"
        "     last_checked_at = NULL"
        " WHERE id = 1"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET last_reviewed_at = '2026-06-23',"
        "     unreviewed_count = 4,"
        "     last_checked_at = NULL"
        " WHERE id = 1"
    )
