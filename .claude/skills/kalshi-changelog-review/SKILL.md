---
name: kalshi-changelog-review
description: Step-by-step procedure for reviewing unreviewed Kalshi API changelog entries and marking them reviewed via an Alembic migration. Use when asked to review the Kalshi changelog, or when kalshi_changelog_state shows unreviewed_count > 0 in the DB, an alert, or the system health UI.
---

# Kalshi changelog review

The system monitors the Kalshi API changelog daily via `freqpred/ingestion/kalshi_changelog.py`. Unreviewed entries surface as alerts and in the system health UI. When asked to review the changelog — or when `unreviewed_count > 0` is noticed — follow this process:

1. **Check the DB for unreviewed entries:**
   ```bash
   docker exec freqpred-db-1 psql -U freqpred -d freqpred -c \
     "SELECT last_reviewed_at, unreviewed_count, has_unreviewed_breaking_change, last_checked_at FROM kalshi_changelog_state WHERE id = 1;"
   ```

2. **Read the live changelog** at `https://docs.kalshi.com/changelog` — fetch all entries published after `last_reviewed_at`.

3. **Audit each unreviewed entry against the codebase.** For each entry, determine whether it requires a code change. Pay particular attention to:
   - Field renames or removals on order, market, or fill responses (the Kalshi client in `freqpred/markets/` parses these)
   - WebSocket channel or event type changes (`freqpred/markets/watcher.py`, `position_watcher.py`)
   - Endpoint URL or base host changes (the External API gotchas section of the root `CLAUDE.md` + any hardcoded URLs)
   - Rate limit scheme changes (retry/backoff logic)
   - Breaking changes flagged in the RSS `<category>` tags
   If a code change is required, implement and test it before proceeding.

4. **Mark entries as reviewed** by creating an Alembic migration that updates all three fields atomically:
   ```bash
   uv run alembic revision -m "mark_kalshi_changelog_reviewed_YYYY_MM_DD"
   ```
   In the migration's `upgrade()`:
   ```python
   op.execute(
       "UPDATE kalshi_changelog_state"
       " SET last_reviewed_at = 'YYYY-MM-DD',"  # literal review date, NOT CURRENT_DATE
       "     unreviewed_count = 0,"
       "     last_checked_at = NULL"
       " WHERE id = 1"
   )
   ```
   **Use the literal review date** (e.g. `'2026-06-04'`), never `CURRENT_DATE`. Migrations are replayed by future developers and CI — `CURRENT_DATE` would stamp the wrong date on any replay, making `last_reviewed_at` meaningless as an audit trail.

   Nulling `last_checked_at` forces the monitor to re-run immediately on next startup and verify the count from RSS. Setting `unreviewed_count = 0` makes the dashboard correct right away without waiting for that re-run.

   Then apply:
   ```bash
   DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred" uv run alembic upgrade head
   ```
