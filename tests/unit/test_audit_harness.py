"""Tests for the point-in-time correctness of the assessor audit harness.

The audit reuses production prompt construction but must feed it the market as
it looked at signal time. Anything that leaks post-resolution state into the
prompt silently biases every arm of every future screen, so the masking is
covered here rather than trusted.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_assessor_enhancement.py"


def _load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location("audit_assessor_enhancement", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()


def _load_freezer_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "freeze_assessor_eval_set",
        Path(__file__).resolve().parents[2] / "scripts" / "freeze_assessor_eval_set.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freezer = _load_freezer_module()


def _payload_with_liquidity() -> dict:
    """A payload shaped like _build_prompt_payload's market_liquidity block,
    carrying the settled-state values a finalized market returns today."""
    return {
        "market": {"market_id": "KXTRUMPSAY-26JUN01-TULS", "days_to_close": 1.4},
        "market_liquidity": {
            "yes_bid": 0.87,
            "yes_ask": 0.88,
            "spread": 0.01,
            "yes_bid_size_dollars": 0.0,
            "yes_ask_size_dollars": 0.0,
            "volume_24h": 0.0,
            "open_interest": 0.0,
            "price_updated_at": "2026-07-24T21:00:00+00:00",
            "note": (
                "Wide spread, very low volume_24h, or small book depth "
                "suggests thin/illiquid pricing — a large edge in this "
                "context is likely artificial."
            ),
        },
    }


class TestMaskUnreconstructibleLiquidity:
    def test_settled_state_fields_are_masked_not_left_as_zero(self) -> None:
        """Book depth, volume, open interest and the price timestamp come off
        the CURRENT MarketRow. Every audited market has since finalized, where
        they settle to zero — so leaving them through tells the judge 'no book,
        no volume' on essentially every signal, which is post-resolution
        information, not a signal-time fact."""
        out = audit._mask_unreconstructible_liquidity(_payload_with_liquidity())
        liq = out["market_liquidity"]
        for key in (
            "yes_bid_size_dollars",
            "yes_ask_size_dollars",
            "volume_24h",
            "open_interest",
            "price_updated_at",
        ):
            assert liq[key] == audit.LIQUIDITY_UNAVAILABLE, key
            assert liq[key] != 0.0

    def test_reconstructed_prices_and_spread_are_preserved(self) -> None:
        """yes_bid/yes_ask ARE genuinely recovered by _reconstruct_prices from
        the signal's own stored fields, so masking them would throw away real
        signal-time information."""
        out = audit._mask_unreconstructible_liquidity(_payload_with_liquidity())
        liq = out["market_liquidity"]
        assert liq["yes_bid"] == 0.87
        assert liq["yes_ask"] == 0.88
        assert liq["spread"] == 0.01

    def test_note_forbids_inferring_thinness_from_absence(self) -> None:
        """The shipped note invites the judge to treat thin depth as evidence
        the edge is artificial. Once the fields are masked, the note must say
        'unknown' rather than leaving the old wording to fire on the sentinel."""
        out = audit._mask_unreconstructible_liquidity(_payload_with_liquidity())
        note = out["market_liquidity"]["note"].lower()
        assert "unknown" in note
        assert "not as zero or thin" in note
        assert "artificial" not in note

    def test_is_idempotent(self) -> None:
        """The masker runs after _fix_days_to_close in the arm loop; a double
        application (e.g. a future refactor calling it per-arm) must not corrupt
        the block."""
        once = audit._mask_unreconstructible_liquidity(_payload_with_liquidity())
        twice = audit._mask_unreconstructible_liquidity(
            audit._mask_unreconstructible_liquidity(_payload_with_liquidity())
        )
        assert once == twice

    def test_missing_or_malformed_liquidity_block_is_a_noop(self) -> None:
        """Never raise inside the arm loop over a payload shape change — a
        crash here would abort a paid run mid-flight."""
        assert audit._mask_unreconstructible_liquidity({}) == {}
        assert audit._mask_unreconstructible_liquidity({"market_liquidity": None}) == {
            "market_liquidity": None
        }

    def test_inconsistent_book_is_masked_not_published_as_crossed(self) -> None:
        """A reconstructed bid above ask is not a market state, it is a
        reconstruction failure: only the traded side is stored directly, and the
        other side is reflected around a mid that was captured on a different poll
        cycle. Publishing it invites the judge to discount the signal for
        degenerate pricing that never existed (observed live on
        KXTRUMPSAY-26JUL06-AUTO, which stores mid 0.35 against ask 0.33)."""
        out = audit._mask_unreconstructible_liquidity(
            _payload_with_liquidity(), book_reliable=False
        )
        liq = out["market_liquidity"]
        for key in ("yes_bid", "yes_ask", "spread"):
            assert liq[key] == audit.BOOK_UNRECONSTRUCTIBLE
        note = liq["note"]
        assert "NOT evidence of a crossed or degenerate" in note
        assert "trade_context" in note

    def test_reliable_book_is_still_published(self) -> None:
        """The mask must not fire on the ~90% of signals that reconstruct fine —
        those prices are real signal-time data and are worth having."""
        out = audit._mask_unreconstructible_liquidity(
            _payload_with_liquidity(), book_reliable=True
        )
        liq = out["market_liquidity"]
        assert liq["yes_bid"] == 0.87
        assert liq["yes_ask"] == 0.88
        assert audit.BOOK_UNRECONSTRUCTIBLE not in liq["note"]

    def test_other_payload_sections_untouched(self) -> None:
        out = audit._mask_unreconstructible_liquidity(_payload_with_liquidity())
        assert out["market"] == {
            "market_id": "KXTRUMPSAY-26JUN01-TULS",
            "days_to_close": 1.4,
        }


def _sig(direction: str, mid: float, ask: float) -> SimpleNamespace:
    """Minimal stand-in carrying the five fields the reconstruction reads."""
    return SimpleNamespace(
        direction=direction,
        market_mid_at_signal=mid,
        market_ask_at_signal=ask,
        estimated_probability=0.6,
        edge=0.2,
    )


class TestBookReconstructionConsistency:
    """Both directions are covered deliberately: YES reflects the bid off the mid
    while NO reflects the ask off a bid derived from (1 - no_ask), so the two
    invert under opposite conditions and a one-sided test would miss half the bug.
    """

    def test_yes_consistent_book_accepted(self) -> None:
        # mid below ask: bid = 2(0.30) - 0.33 = 0.27 <= 0.33
        assert audit._book_reconstruction_is_consistent(_sig("YES", 0.30, 0.33)) is True

    def test_yes_inverted_book_rejected(self) -> None:
        # Real case, KXTRUMPSAY-26JUL06-AUTO: a YES mid cannot exceed its own ask.
        # bid = 2(0.35) - 0.33 = 0.37 > 0.33
        assert audit._book_reconstruction_is_consistent(_sig("YES", 0.35, 0.33)) is False

    def test_no_consistent_book_accepted(self) -> None:
        # no_ask 0.30 -> yes_bid 0.70; mid 0.72 -> yes_ask 0.74 >= 0.70
        assert audit._book_reconstruction_is_consistent(_sig("NO", 0.72, 0.30)) is True

    def test_no_inverted_book_rejected(self) -> None:
        # Real case, KXTRUMPSAY-26JUN01-TULS: no_ask 0.13 -> yes_bid 0.87;
        # mid 0.73 -> yes_ask 0.59, a -0.28 crossed spread.
        assert audit._book_reconstruction_is_consistent(_sig("NO", 0.73, 0.13)) is False

    def test_unreconstructible_signal_is_treated_as_unreliable(self) -> None:
        """A missing ask raises inside _reconstruct_prices; that is a failure to
        reconstruct, so it must not be reported as a usable book."""
        assert audit._book_reconstruction_is_consistent(_sig("YES", 0.30, None)) is False


def _pool() -> pd.DataFrame:
    """Resolved-history pool shaped like the audit's calib_df."""
    import datetime as _dt

    rows = []
    old = _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)
    # NO/15-40 wins often and cheaply; YES/15-40 loses. 40 rows each clears MIN_PRIOR_CELL.
    for _ in range(40):
        rows.append({"direction": "NO", "edge_band": "15-40", "close_time": old,
                     "hit": 1.0, "market_p_side": 0.60})
        rows.append({"direction": "YES", "edge_band": "15-40", "close_time": old,
                     "hit": 0.0, "market_p_side": 0.60})
    return pd.DataFrame(rows)


class TestPitBaselinePrior:
    """The baseline is the bar the LLM must clear, so a wrong baseline silently
    flatters or damns every future package. Both directions are covered because
    the whole point of the prior is that NO and YES behave differently."""

    def _sig(self, direction: str):
        return SimpleNamespace(direction=direction, edge=0.25)

    def test_hit_rate_prior_separates_directions(self) -> None:
        import datetime as _dt

        as_of = _dt.datetime(2026, 6, 1, tzinfo=_dt.UTC)
        no = audit._pit_baseline_prior(_pool(), self._sig("NO"), as_of)
        yes = audit._pit_baseline_prior(_pool(), self._sig("YES"), as_of)
        assert no == pytest.approx(1.0)
        assert yes == pytest.approx(0.0)
        assert no > yes

    def test_profit_edge_prior_is_net_of_price(self) -> None:
        """Hit rate alone is meaningless without the price paid: a NO bought at
        0.60 that wins 100% earns +0.40, not +1.00."""
        import datetime as _dt

        as_of = _dt.datetime(2026, 6, 1, tzinfo=_dt.UTC)
        no = audit._pit_baseline_profit_edge(_pool(), self._sig("NO"), as_of)
        yes = audit._pit_baseline_profit_edge(_pool(), self._sig("YES"), as_of)
        assert no == pytest.approx(0.40)
        assert yes == pytest.approx(-0.60)

    def test_is_point_in_time(self) -> None:
        """Only history that had already closed may inform the prior."""
        import datetime as _dt

        before_any = _dt.datetime(2025, 1, 1, tzinfo=_dt.UTC)
        assert audit._pit_baseline_prior(_pool(), self._sig("NO"), before_any) == 0.5
        assert audit._pit_baseline_profit_edge(_pool(), self._sig("NO"), before_any) == 0.0

    def test_thin_cell_falls_back_rather_than_returning_noise(self) -> None:
        """A cell below MIN_PRIOR_CELL must widen to direction, then global —
        never return a base rate computed from a handful of rows."""
        import datetime as _dt

        pool = _pool()
        pool.loc[pool.edge_band == "15-40", "edge_band"] = "0-15"  # no >40 cell exists
        val = audit._pit_baseline_prior(pool, SimpleNamespace(direction="NO", edge=0.90),
                                        _dt.datetime(2026, 6, 1, tzinfo=_dt.UTC))
        assert val == pytest.approx(1.0)  # fell back to direction-level NO rows


class TestV8ProfitEdgeFraming:
    """v8's core change: point the judge at profit-vs-price, not at the model's
    self-consistency. The two diverge on 8 of 34 calibration cells covering 2,880
    signals, and where they conflict the shipped wording sizes DOWN profitable
    bands."""

    def _calib(self) -> dict:
        # Real shape from KXTRUMPSAY/NO/>40: profitable, yet wildly "overconfident".
        return {
            "this_signal_edge_band": ">40",
            "all_directions": {
                "n_signals": 355, "n_markets": 80,
                "hit_rate": 0.245, "avg_market_implied_p": 0.261,
                "avg_model_implied_p": 0.852,
            },
            "same_direction_only": {
                "n_signals": 153, "n_markets": 40,
                "hit_rate": 0.281, "avg_market_implied_p": 0.238,
                "avg_model_implied_p": 0.839,
            },
        }

    def test_profit_edge_is_computed_from_the_price_not_the_model_claim(self) -> None:
        out = audit._add_profit_edge(self._calib())
        same = out["same_direction_only"]
        # +4.3pp over the price paid: profitable.
        assert same["profit_edge_vs_price"] == pytest.approx(0.281 - 0.238, abs=1e-4)
        # ...while the overconfidence gap is -0.558. Opposite signs, same cell.
        assert same["hit_rate"] - same["avg_model_implied_p"] < -0.5
        assert same["profit_edge_vs_price"] > 0

    def test_description_demotes_the_overconfidence_gap(self) -> None:
        out = audit._add_profit_edge(self._calib())
        d = out["description"]
        assert "profit_edge_vs_price" in d
        assert "self-knowledge" in d
        assert "points the other way" in d

    def test_handles_absent_or_empty_cells(self) -> None:
        assert audit._add_profit_edge(None) is None
        thin = {"same_direction_only": {"n_signals": 0, "n_markets": 0}}
        assert "profit_edge_vs_price" not in audit._add_profit_edge(thin)["same_direction_only"]


class TestVersionCohort:
    """Historical performance is strongly signal-prompt-version dependent
    (KXTRUMPSAY NO profit edge: -0.240 on signal-v7, +0.133 on signal-v11), so
    pooling versions describes a model production no longer runs."""

    def _pool(self, n_v11: int, n_old: int) -> pd.DataFrame:
        rows = [{"prompt_version": "signal-v11", "hit": 1.0} for _ in range(n_v11)]
        rows += [{"prompt_version": "signal-v4", "hit": 0.0} for _ in range(n_old)]
        return pd.DataFrame(rows)

    def test_filters_to_the_signals_own_version_when_cohort_is_adequate(self) -> None:
        pool = self._pool(audit.MIN_VERSION_COHORT, 500)
        cohort, label = audit._version_cohort(pool, SimpleNamespace(prompt_version="signal-v11"))
        assert label == "signal-v11"
        assert set(cohort.prompt_version) == {"signal-v11"}

    def test_falls_back_to_full_pool_on_a_freshly_shipped_version(self) -> None:
        """A new prompt version must not start from zero history."""
        pool = self._pool(5, 500)
        cohort, label = audit._version_cohort(pool, SimpleNamespace(prompt_version="signal-v11"))
        assert label == "all_versions_fallback"
        assert len(cohort) == 505

    def test_fallback_label_is_surfaced_so_the_judge_knows(self) -> None:
        _, label = audit._version_cohort(
            self._pool(0, 500), SimpleNamespace(prompt_version="signal-v99")
        )
        assert "fallback" in label


class TestChallengerIsDisarmedAfterAdoption:
    """After an adoption the challenger MUST be undefined. Leaving CHALLENGER_*
    pointing at the just-adopted package makes a re-run measure
    current-vs-current — a guaranteed null result billed at real API rates
    (~$0.85 for a challenger arm on `z-ai/glm-5.2` over the 76-signal frozen
    set; it was ~$3.30 when the incumbent was Opus). Most recently armed for
    the 2026-08-09 opus-5 -> z-ai/glm-5.2 model swap, then disarmed on
    adoption."""

    def test_challenger_hooks_are_unset(self) -> None:
        assert audit.CHALLENGER_VERSION is None
        assert audit.CHALLENGER_SYSTEM_PROMPT is None
        assert audit.CHALLENGER_MODEL is None

    def test_payload_builder_refuses_to_run(self) -> None:
        import asyncio

        with pytest.raises(NotImplementedError):
            asyncio.run(audit._challenger_payload(None, None, {}, None))

    def test_adopted_prompt_is_retained_as_a_record(self) -> None:
        """The screened prompt stays available to diff a future proposal
        against — just not wired to the live hook."""
        assert "profit_edge_vs_price" in audit.ADOPTED_V8_SYSTEM_PROMPT
        assert audit.ADOPTED_V8_SYSTEM_PROMPT is not audit.CHALLENGER_SYSTEM_PROMPT


class TestFrozenSetCacheKey:
    """The freezer harvests already-paid current-arm responses from llm_queries.
    Reuse is only sound if the stored response came from byte-identical input, so
    the freezer records a hash of the STORED PROMPT and the runner compares it to
    the freshly-rendered payload hash. Before this fix the freezer never wrote
    `cached_hash` at all, so the runner's check compared against a missing field
    and reuse was silently, permanently disabled."""

    def test_hash_is_over_exact_bytes(self) -> None:
        import hashlib

        text = '{"a": 1}'
        expected = hashlib.sha256(text.encode()).hexdigest()[:16]
        assert freezer._hash_text(text) == expected

    def test_payload_hash_matches_hashing_the_rendered_prompt(self) -> None:
        """The whole scheme rests on this: a payload hashed as a dict must equal
        the prompt string hashed as text, because every producer renders with
        json.dumps(..., indent=2, sort_keys=True). If those ever diverge, cached
        responses stop matching and the harness silently re-pays for them."""
        payload = {"z": 1, "a": {"nested": True}}
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        assert freezer._payload_hash(payload) == freezer._hash_text(rendered)

    def test_key_ordering_does_not_change_the_hash(self) -> None:
        assert freezer._payload_hash({"a": 1, "b": 2}) == freezer._payload_hash(
            {"b": 2, "a": 1}
        )

    def test_any_content_change_changes_the_hash(self) -> None:
        """A harness change that alters the payload must invalidate the cache
        rather than pair an old score with a new prompt."""
        assert freezer._payload_hash({"a": 1}) != freezer._payload_hash({"a": 2})


@pytest.mark.parametrize("arm", ["current", "challenger"])
def test_both_arms_are_declared(arm: str) -> None:
    assert arm in audit.ARM_NAMES
