"""mark_kalshi_changelog_reviewed_2026_06_02

Revision ID: 0042
Revises: 0041
Create Date: 2026-06-02 17:44:19.717047

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0042'
down_revision: Union[str, None] = '0041'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET last_reviewed_at = CURRENT_DATE,"
        "     unreviewed_count = 0,"
        "     last_checked_at = NULL"
        " WHERE id = 1"
    )


def downgrade() -> None:
    pass
