"""Signal pipeline: RAG retrieval → hash check → LLM analysis → Signal."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.ingestion.models import CatalystQueryRow, CatalystRunRow
from freqpred.llm.client import LLMClient, LLMError
from freqpred.markets.models import Market, MarketRow
from freqpred.metrics.series_history import get_series_history_for_market
from freqpred.rag.models import Document, DocumentMarketLinkRow
from freqpred.rag.retriever import Embedder, compute_retrieval_hash, retrieve
from freqpred.signal.cache import scheduled_cooldown_remaining, should_skip, should_skip_scheduled
from freqpred.signal.llm import (
    PROMPT_VERSION,
    SIGNAL_ANALYSIS_TOOL,
    SYSTEM_PROMPT,
    build_prompt,
    parse_signal_response,
)
from freqpred.signal.models import Signal, SignalRow

log = structlog.get_logger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_TOP_K = 10


def compute_signal_edge(
    direction: str,
    estimated_probability: float,
    yes_bid: float,
    yes_ask: float,
) -> tuple[float, float | None]:
    """Return ``(edge, market_ask_at_signal)`` for a signal's side.

    Edge uses the actual ask price for the signal's side — not mid_price —
    because mid on illiquid markets is meaningless.
    YES: edge = prob - yes_ask  (we think YES is underpriced vs what it costs)
    NO:  edge = (1-prob) - no_ask = (1-prob) - (1-yes_bid) = yes_bid - prob
    SKIP: edge = prob - yes_ask for audit; no_ask not applicable (ask is None).

    Pure function — shared by signal creation, price-move repricing, and the
    replay harness so all three always price edge identically.
    """
    if direction == "NO":
        market_ask_at_signal: float | None = round(1.0 - yes_bid, 4)
        edge = (1.0 - estimated_probability) - market_ask_at_signal
    elif direction == "YES":
        market_ask_at_signal = yes_ask
        edge = estimated_probability - market_ask_at_signal
    else:  # SKIP
        market_ask_at_signal = None
        edge = estimated_probability - yes_ask
    return edge, market_ask_at_signal


class SignalPipeline:
    """Orchestrates RAG retrieval, hash deduplication, and LLM probability analysis.

    Args:
        session_factory:  SQLAlchemy async session factory.
        embedder:         Voyage AI embedder (or any ``Embedder`` protocol impl).
        llm_client:       Auditing LLM wrapper — every call is logged automatically.
        model:            Anthropic model ID for analysis (default: claude-sonnet-4-6).
        top_k:            Number of documents to retrieve per market (default: 10).
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        llm_client: LLMClient,
        model: str = _DEFAULT_MODEL,
        top_k: int = _TOP_K,
        factbase_series_allowlist: frozenset[str] = frozenset(),
        max_scheduled_interval_hours: float = 24.0,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._llm_client = llm_client
        self._model = model
        self._top_k = top_k
        self._factbase_series_allowlist = factbase_series_allowlist
        self._max_scheduled_interval_hours = max_scheduled_interval_hours

    async def analyze(
        self,
        market: Market,
        trigger: str = "scheduled",
        force: bool = False,
    ) -> Signal | None:
        """Analyze a market and return a new Signal.

        Steps:
        1. Retrieve top-K documents via RAG.
        2. If no documents retrieved → return None (no evidence to analyze).
        3. Compute retrieval hash from doc IDs.
        4. If hash == current signal hash and trigger is not "scheduled" → return None.
           Skipped when *force* is True. Scheduled runs always call the LLM because
           context beyond raw documents (series history, FactBase counts) can shift
           the probability even when the doc set is unchanged.
        5. Call Claude with market question + docs as context.
        6. Parse JSON response.
        7. Write Signal + DocumentMarketLinks + update Market.current_signal_id
           in a single transaction.
        8. Return Signal.

        Args:
            force: If True, bypass the retrieval-hash deduplication check and
                   always call the LLM regardless of whether evidence has changed.

        Returns:
            A new ``Signal`` if analysis succeeded.
            ``None`` if no documents exist, the LLM call fails, or the response
            is malformed.
        """
        async with self._session_factory() as session:
            # Load active catalyst queries for this market so the retriever can
            # supplement the market-question core set with catalyst-driven docs.
            cat_result = await session.execute(
                select(CatalystQueryRow)
                .join(CatalystRunRow, CatalystQueryRow.run_id == CatalystRunRow.id)
                .where(
                    CatalystRunRow.market_id == market.id,
                    CatalystRunRow.is_active.is_(True),
                )
            )
            catalyst_query_texts = [row.query_text for row in cat_result.scalars().all()]

            # Step 1: retrieve relevant documents with cosine similarity scores
            doc_pairs = await retrieve(
                session,
                self._embedder,
                market.question,
                market.id,
                top_k=self._top_k,
                catalyst_queries=catalyst_query_texts or None,
            )

            # Step 2: skip if no documents were retrieved (unless forced)
            if not doc_pairs:
                if not force:
                    log.debug(
                        "signal.pipeline.skip_no_docs",
                        market_id=market.id,
                    )
                    return None
                log.info(
                    "signal.pipeline.force_no_docs",
                    market_id=market.id,
                )

            docs = [doc for doc, _ in doc_pairs]

            # Step 3: compute retrieval hash from returned doc IDs
            doc_ids = [d.id for d in docs]
            new_hash = compute_retrieval_hash(doc_ids)

            # Step 5: skip if there is nothing new to analyse.
            # - Non-scheduled: skip when the RAG doc set hasn't changed (hash match).
            # - Scheduled: skip when the doc set AND FactBase are both unchanged AND
            #   the minimum rerun interval hasn't elapsed yet.  This lets us react
            #   immediately to new evidence while still guaranteeing a temporal-
            #   reasoning rerun at least once every max_scheduled_interval_hours.
            _skip: bool
            if trigger == "scheduled":
                _skip = await should_skip_scheduled(
                    session, market.id, new_hash, self._max_scheduled_interval_hours
                )
            else:
                _skip = await should_skip(session, market.current_signal_id, new_hash)
            if not force and _skip:
                # Evidence unchanged — but if price moved we can still create a
                # new signal by cloning the current one at the new price (no LLM call).
                cloned = await self._clone_at_price(session, market)
                if cloned is not None:
                    await session.commit()
                    log.info(
                        "signal.pipeline.price_reprice",
                        market_id=market.id,
                        signal_id=cloned.id,
                        new_mid=market.mid_price,
                        edge=cloned.edge,
                    )
                    return cloned
                log.debug(
                    "signal.pipeline.skip_unchanged",
                    market_id=market.id,
                    retrieval_hash=new_hash,
                )
                return None

            # Step 5b: low-confidence cooldown — skip LLM for scheduled analyses
            # when the last scheduled signal was below the confidence threshold
            # and was created recently.  Price-moved clones are still allowed.
            if trigger == "scheduled" and not force:
                cooldown_h = await scheduled_cooldown_remaining(session, market.id)
                if cooldown_h > 0:
                    log.debug(
                        "signal.pipeline.cooldown_skip",
                        market_id=market.id,
                        cooldown_hours_remaining=round(cooldown_h, 1),
                    )
                    cloned = await self._clone_at_price(session, market)
                    if cloned is not None:
                        await session.commit()
                        log.info(
                            "signal.pipeline.price_reprice",
                            market_id=market.id,
                            signal_id=cloned.id,
                            new_mid=market.mid_price,
                            edge=cloned.edge,
                        )
                        return cloned
                    return None

            # Step 6: build prompt and call LLM
            series_history = None
            if market.series_ticker:
                option_code = market.id.rsplit("-", 1)[-1] if "-" in market.id else market.id
                series_history = await get_series_history_for_market(
                    session, market.series_ticker, option_code
                )

            phrase_data = None
            if market.series_ticker and market.series_ticker in self._factbase_series_allowlist:
                from freqpred.ingestion.fetchers.factbase import phrase_row_to_data
                from freqpred.ingestion.models import FactbasePhraseRow
                fb_result = await session.execute(
                    select(FactbasePhraseRow).where(FactbasePhraseRow.market_id == market.id)
                )
                fb_row = fb_result.scalar_one_or_none()
                if fb_row is not None:
                    phrase_data = phrase_row_to_data(fb_row)

            prompt = build_prompt(market, docs, series_history=series_history, phrase_data=phrase_data)
            try:
                llm_response = await self._llm_client.complete(
                    prompt,
                    self._model,
                    query_type="market_analysis",
                    system=SYSTEM_PROMPT,
                    cache_system=True,
                    market_id=market.id,
                    max_tokens=1024,
                    json_tool=SIGNAL_ANALYSIS_TOOL,
                )
            except LLMError as exc:
                log.error(
                    "signal.pipeline.llm_error",
                    market_id=market.id,
                    error=str(exc),
                )
                return None

            # Step 5: parse structured JSON response
            parsed = parse_signal_response(llm_response.content)
            if parsed is None:
                log.error(
                    "signal.pipeline.parse_failed",
                    market_id=market.id,
                    content_preview=llm_response.content[:200],
                )
                return None

            # Monitoring: log prior→posterior delta for calibration health checks
            _prior = parsed.get("prior")
            _posterior = parsed.get("posterior")
            _update_count = len(parsed.get("updates_applied") or [])
            if _prior is not None and _posterior is not None:
                _delta = round(abs(_posterior - _prior), 4)
                log.info(
                    "signal.prior_posterior",
                    market_id=str(market.id),
                    prior=_prior,
                    posterior=_posterior,
                    delta=_delta,
                    update_count=_update_count,
                    flagged_underjustified=_delta > 0.15 and _update_count == 0,
                )

            # Step 6: write signal + document links + update market atomically
            signal = await self._write_signal(
                session=session,
                market=market,
                doc_pairs=doc_pairs,
                parsed=parsed,
                retrieval_hash=new_hash,
                raw_context=prompt,
                trigger=trigger,
                llm_query_id=llm_response.llm_query_id,
            )
            await session.commit()

        if signal.direction == "YES":
            side_price = market.yes_ask
        elif signal.direction == "NO":
            side_price = round(1.0 - market.yes_bid, 4)
        else:
            side_price = None
        log.info(
            "signal.pipeline.new_signal",
            market_id=market.id,
            signal_id=signal.id,
            probability=signal.estimated_probability,
            direction=signal.direction,
            side_price=side_price,
        )
        return signal

    async def _write_signal(
        self,
        session: AsyncSession,
        market: Market,
        doc_pairs: list[tuple[Document, float]],
        parsed: dict,
        retrieval_hash: str,
        raw_context: str,
        trigger: str,
        llm_query_id: int | None = None,
    ) -> Signal:
        """Insert Signal, DocumentMarketLinks, and update Market.current_signal_id.

        All writes use the caller's session; the caller is responsible for commit.
        """
        now = datetime.now(UTC)
        signal_id = uuid.uuid4()
        estimated_probability = parsed["probability"]
        direction = parsed["direction"]
        edge, market_ask_at_signal = compute_signal_edge(
            direction, estimated_probability, market.yes_bid, market.yes_ask
        )
        docs = [doc for doc, _ in doc_pairs]

        signal_row = SignalRow(
            id=signal_id,
            market_id=market.id,
            estimated_probability=estimated_probability,
            confidence=parsed["confidence"],
            edge=edge,
            market_mid_at_signal=market.mid_price,
            market_ask_at_signal=market_ask_at_signal,
            direction=direction,
            reasoning=parsed["reasoning"],
            sources=[d.id for d in docs],
            retrieval_hash=retrieval_hash,
            model_used=self._model,
            prompt_version=PROMPT_VERSION,
            trigger=trigger,
            raw_context=raw_context,
            llm_query_id=llm_query_id,
        )
        session.add(signal_row)

        # Flush so the signal PK exists in the DB before DocumentMarketLinkRow FKs reference it
        await session.flush()

        # Create one DocumentMarketLink per retrieved document using the actual cosine similarity
        for doc, similarity_score in doc_pairs:
            link = DocumentMarketLinkRow(
                document_id=uuid.UUID(doc.id),
                market_id=market.id,
                signal_id=signal_id,
                relevance_score=similarity_score,
                linked_at=now,
            )
            session.add(link)

        # Update Market.current_signal_id to point at the new signal
        await session.execute(
            update(MarketRow)
            .where(MarketRow.id == market.id)
            .values(current_signal_id=signal_id)
        )

        return Signal(
            id=str(signal_id),
            market_id=market.id,
            estimated_probability=estimated_probability,
            confidence=parsed["confidence"],
            edge=edge,
            market_mid_at_signal=market.mid_price,
            market_ask_at_signal=market_ask_at_signal,
            direction=parsed["direction"],
            reasoning=parsed["reasoning"],
            sources=[d.id for d in docs],
            retrieval_hash=retrieval_hash,
            model_used=self._model,
            prompt_version=PROMPT_VERSION,
            trigger=trigger,
            created_at=now,
            raw_context=raw_context,
        )

    async def _clone_at_price(
        self,
        session: AsyncSession,
        market: Market,
        price_move_threshold: float = 0.05,
    ) -> Signal | None:
        """Clone the current signal at the new market price without calling the LLM.

        Returns a new Signal if price has moved more than *price_move_threshold*
        since the current signal, otherwise returns None (no action needed).
        The new signal has the same probability/confidence/reasoning but
        recalculated edge and updated market_mid_at_signal.
        """
        if market.current_signal_id is None:
            return None

        try:
            signal_uuid = uuid.UUID(str(market.current_signal_id))
        except (ValueError, AttributeError):
            return None

        result = await session.execute(
            select(SignalRow).where(SignalRow.id == signal_uuid)
        )
        current = result.scalar_one_or_none()
        if current is None:
            return None

        if abs(market.mid_price - current.market_mid_at_signal) <= price_move_threshold:
            return None

        now = datetime.now(UTC)
        new_id = uuid.uuid4()

        new_edge, new_ask = compute_signal_edge(
            current.direction, current.estimated_probability, market.yes_bid, market.yes_ask
        )

        signal_row = SignalRow(
            id=new_id,
            market_id=market.id,
            estimated_probability=current.estimated_probability,
            confidence=current.confidence,
            edge=new_edge,
            market_mid_at_signal=market.mid_price,
            market_ask_at_signal=new_ask,
            direction=current.direction,
            reasoning=current.reasoning,
            sources=current.sources,
            retrieval_hash=current.retrieval_hash,
            model_used=current.model_used,
            prompt_version=current.prompt_version,
            trigger="price_moved",
            raw_context=current.raw_context,
        )
        session.add(signal_row)
        await session.flush()

        await session.execute(
            update(MarketRow)
            .where(MarketRow.id == market.id)
            .values(current_signal_id=new_id)
        )

        return Signal(
            id=str(new_id),
            market_id=market.id,
            estimated_probability=current.estimated_probability,
            confidence=current.confidence,
            edge=new_edge,
            market_mid_at_signal=market.mid_price,
            market_ask_at_signal=new_ask,
            direction=current.direction,
            reasoning=current.reasoning,
            sources=list(current.sources),
            retrieval_hash=current.retrieval_hash,
            model_used=current.model_used,
            prompt_version=current.prompt_version,
            trigger="price_moved",
            created_at=now,
            raw_context=current.raw_context,
        )
