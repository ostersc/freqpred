"""mark_kalshi_changelog_reviewed_2026_07_04

Revision ID: 0051
Revises: 0050
Create Date: 2026-07-04 12:40:18.210968

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0051'
down_revision: str | None = '0050'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET last_reviewed_at = '2026-07-04',"
        "     unreviewed_count = 0,"
        "     last_checked_at = NULL"
        " WHERE id = 1"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET last_reviewed_at = '2026-07-02',"
        "     unreviewed_count = 1,"
        "     last_checked_at = NULL"
        " WHERE id = 1"
    )
