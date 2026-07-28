"""unpoison_future_candle_cursors

Repairs candle_fetch_cursors rows that claim coverage into the future.

A `freqpred candles backfill` run on 2026-07-27 21:43 swept every market matching
its selector, including 32 that were still trading, and recorded each one's
coverage as running to its close_time. For an open market that end is in the
future, so the row claims coverage of candles that did not exist yet.

`refresh_recent_candles` skips any market whose cursor already reaches its
close_time, which makes the false claim permanent: those markets would never be
fetched again, not even after they closed and their candles finally existed.
KXTRUMPSAY-26AUG03-TIKT was left holding ten candles for its entire lifetime.

Clamping covered_to back to last_fetched_at restores the truth — coverage ends
where the fetch ended — and lets the daily refresh complete each market's history
once it closes. `_record_cursor` now applies the same clamp on write, so this is
a one-time repair rather than something that can recur.

Only rows whose coverage runs past their own fetch time are touched; correctly
recorded rows (every closed market) are left alone.

Revision ID: 0062
Revises: 0061
Create Date: 2026-07-28 00:00:00.000000

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0062'
down_revision: str | None = '0061'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "UPDATE candle_fetch_cursors"
        " SET covered_to = GREATEST(last_fetched_at, covered_from)"
        " WHERE covered_to > last_fetched_at"
    )


def downgrade() -> None:
    # The pre-repair values were fabricated (a future close_time that was never
    # fetched), so there is nothing faithful to restore. Re-widening the window
    # would only reinstate the bug. Coverage that is understated is corrected on
    # the next refresh cycle at the cost of one request per market, so leaving
    # the repaired values in place is both safe and self-healing.
    pass
