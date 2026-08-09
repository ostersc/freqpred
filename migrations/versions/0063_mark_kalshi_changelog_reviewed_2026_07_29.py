"""mark_kalshi_changelog_reviewed_2026_07_29

Revision ID: 0063
Revises: 0062
Create Date: 2026-07-29 00:00:00.000000

Reviewed the 8 entries published 2026-07-28..2026-07-29 (docs page groups them
under the July 28 / July 30 / August 6 headers).  All are additive or apply to
API surfaces this system does not use — no code change required:

- Richer combo-validation errors on multivariate market creation: we never POST
  to /multivariate_event_collections.
- ``service`` deprecated then removed from REST error bodies: KalshiAPIError
  stores ``resp.text`` verbatim and callers branch on ``status_code``; the only
  body substring check is on the ``code`` value ``insufficient_scope``.
- ``exchange_index`` on market_lifecycle_v2 / GET /series: lifecycle messages are
  parsed with dict ``.get()`` and Kalshi Pydantic schemas use the default
  ``extra="ignore"``.
- New GET /live_data/events/{event_ticker}: unused; serves crypto/commodity/
  weather data.
- Subaccount-restricted keys on order queue position endpoints: we use neither
  subaccounts nor the queue position endpoints.
- ``cadence`` in event ``product_metadata``: we do not parse product_metadata.

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0063'
down_revision: str | None = '0062'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET last_reviewed_at = '2026-07-29',"
        "     unreviewed_count = 0,"
        "     last_checked_at = NULL"
        " WHERE id = 1"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET last_reviewed_at = '2026-07-27',"
        "     unreviewed_count = 0,"
        "     last_checked_at = NULL"
        " WHERE id = 1"
    )
