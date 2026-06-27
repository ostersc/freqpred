"""mark_kalshi_changelog_reviewed_2026_07_02

Revision ID: 0050
Revises: 0049
Create Date: 2026-06-24 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0050'
down_revision: Union[str, None] = '0049'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


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
