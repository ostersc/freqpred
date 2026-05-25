"""Mark Kalshi changelog reviewed through 2026-05-25.

Reviewed entry: Fixed-point dollars added to GET /portfolio/balance (May 24, 2026).
Additive change — balance_dollars field added alongside existing balance (cents).
No code change required; existing balance field continues to be returned.

Revision ID: 0039
Revises: 0038
Create Date: 2026-05-25
"""
from alembic import op

revision = "0039"
down_revision = "0038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state SET last_reviewed_at = '2026-05-25' WHERE id = 1"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state SET last_reviewed_at = '2026-05-21' WHERE id = 1"
    )
