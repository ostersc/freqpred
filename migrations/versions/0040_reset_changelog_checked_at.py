"""Reset unreviewed_count and last_checked_at after changelog review on 2026-05-25.

Sets unreviewed_count = 0 (all entries reviewed) and nulls last_checked_at so
the monitor re-runs immediately on next startup to verify the count from RSS.

Revision ID: 0040
Revises: 0039
Create Date: 2026-05-25
"""
from alembic import op

revision = "0040"
down_revision = "0039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET unreviewed_count = 0, last_checked_at = NULL"
        " WHERE id = 1"
    )


def downgrade() -> None:
    pass
