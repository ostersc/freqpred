"""mark_kalshi_changelog_reviewed_2026_08_07

Revision ID: 0064
Revises: 0063
Create Date: 2026-08-08 00:00:00.000000

Reviewed the 12 entries published 2026-07-30..2026-08-07 (docs page groups them
under the August 6 / August 13 / August 17 headers).  All are additive or apply
to API surfaces this system does not use — no code change required:

- Centicent pricing on multivariate (combo) markets
  (``center_centi_edge_centi_cent``, $0.0001 uniform tick): we do not trade
  combo markets, and both the REST and WebSocket paths already read prices from
  the ``*_dollars`` string fields (``KalshiMarketSchema`` properties,
  ``PositionWatcher`` ticker handling) and round to 4 decimals, so sub-cent
  prices survive.  ``round_to_tick`` snaps to a whole cent, which stays a valid
  price under the new structure (0.01 / 0.0001 = 100).  Docstring in
  ``freqpred/markets/ticks.py`` updated to name the new structure.
- Multivariate lookup endpoint and ``multivariate`` WebSocket channel removed:
  we subscribe only to ``ticker``, ``market_lifecycle_v2``, ``user_orders`` and
  ``fill``, and never call the multivariate REST surface.
- ``is_block_trade`` on WebSocket trade messages: we do not subscribe to the
  ``trade`` channel.
- ``description`` on ``exchange_index_statuses`` entries: additive; Kalshi
  Pydantic schemas use the default ``extra="ignore"``.
- Margin changes (order groups bound to a single ``exchange_index``, sided
  ``long_leverage_estimates``/``short_leverage_estimates``): we do not use the
  margin API.
- Order group changes (limit raised 25,000 → 100,000, ``subaccount`` parameter
  on the limit endpoint): we do not use order groups or subaccounts.
- Intra-exchange transfer history endpoints: new, unused.
- FIX changes (``LastMkt<30>`` on ExecutionReport, richer combo-validation
  fields on ``QuoteRequestReject``): we do not use the FIX gateway.

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0064'
down_revision: str | None = '0063'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET last_reviewed_at = '2026-08-07',"
        "     unreviewed_count = 0,"
        "     last_checked_at = NULL"
        " WHERE id = 1"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE kalshi_changelog_state"
        " SET last_reviewed_at = '2026-07-29',"
        "     unreviewed_count = 0,"
        "     last_checked_at = NULL"
        " WHERE id = 1"
    )
