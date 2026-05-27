"""Mark Kalshi changelog reviewed through 2026-05-27.

Reviewed entry: Fractional quantities for RFQs (published May 26, 2026; effective June 4, 2026).
New Feature / Upcoming — adds fractional contract quantities to RFQ endpoints
(POST /communications/rfqs, GET /communications/rfqs, GET /communications/quotes)
and FIX QuoteRequest/Quote/QuoteStatusReport messages.
freqpred does not use RFQ or FIX endpoints; no code change required.

Revision ID: 0041
Revises: 0040
Create Date: 2026-05-27
"""
from alembic import op

revision = "0041"
down_revision = "0040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET last_reviewed_at = CURRENT_DATE,"
        "     unreviewed_count = 0,"
        "     has_unreviewed_breaking_change = false,"
        "     last_checked_at = NULL"
        " WHERE id = 1"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET last_reviewed_at = '2026-05-25',"
        "     unreviewed_count = 1,"
        "     has_unreviewed_breaking_change = false,"
        "     last_checked_at = NULL"
        " WHERE id = 1"
    )
