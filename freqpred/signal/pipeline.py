"""Signal pipeline: RAG retrieval → hash check → LLM analysis → Signal."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.llm.client import LLMClient, LLMError
from freqpred.markets.models import Market, MarketRow
from freqpred.rag.models import Document, DocumentMarketLinkRow
from freqpred.rag.retriever import Embedder, compute_retrieval_hash, retrieve
from freqpred.signal.cache import should_skip
from freqpred.signal.llm import PROMPT_VERSION, SYSTEM_PROMPT, build_prompt, parse_signal_response
from freqpred.signal.models import Signal, SignalRow

log = structlog.get_logger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_TOP_K = 10


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
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._llm_client = llm_client
        self._model = model
        self._top_k = top_k

    async def analyze(
        self,
        market: Market,
        trigger: str = "scheduled",
    ) -> Signal | None:
        """Analyze a market and return a new Signal if evidence has changed.

        Steps:
        1. Retrieve top-K documents via RAG.
        2. If no documents retrieved → return None (no evidence to analyze).
        3. Compute retrieval hash from doc IDs.
        4. If hash == current signal hash → return None (no new evidence).
        5. Call Claude with market question + docs as context.
        6. Parse JSON response.
        7. Write Signal + DocumentMarketLinks + update Market.current_signal_id
           in a single transaction.
        8. Return Signal.

        Returns:
            A new ``Signal`` if evidence has changed and LLM analysis succeeded.
            ``None`` if evidence is unchanged, the LLM call fails, or the
            response is malformed.
        """
        async with self._session_factory() as session:
            # Step 1: retrieve relevant documents with cosine similarity scores
            doc_pairs = await retrieve(
                session,
                self._embedder,
                market.question,
                market.id,
                top_k=self._top_k,
            )

            # Step 2: skip if no documents were retrieved
            if not doc_pairs:
                log.debug(
                    "signal.pipeline.skip_no_docs",
                    market_id=market.id,
                )
                return None

            docs = [doc for doc, _ in doc_pairs]

            # Step 3: compute retrieval hash from returned doc IDs
            doc_ids = [d.id for d in docs]
            new_hash = compute_retrieval_hash(doc_ids)

            # Step 5: skip if evidence unchanged since last signal
            if await should_skip(session, market.current_signal_id, new_hash):
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

            # Step 6: build prompt and call LLM
            prompt = build_prompt(market, docs)
            try:
                llm_response = await self._llm_client.complete(
                    prompt,
                    self._model,
                    query_type="market_analysis",
                    system=SYSTEM_PROMPT,
                    market_id=market.id,
                    max_tokens=1024,
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

            # Step 6: write signal + document links + update market atomically
            signal = await self._write_signal(
                session=session,
                market=market,
                doc_pairs=doc_pairs,
                parsed=parsed,
                retrieval_hash=new_hash,
                raw_context=prompt,
                trigger=trigger,
            )
            await session.commit()

        log.info(
            "signal.pipeline.new_signal",
            market_id=market.id,
            signal_id=signal.id,
            probability=signal.estimated_probability,
            direction=signal.direction,
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
    ) -> Signal:
        """Insert Signal, DocumentMarketLinks, and update Market.current_signal_id.

        All writes use the caller's session; the caller is responsible for commit.
        """
        now = datetime.now(timezone.utc)
        signal_id = uuid.uuid4()
        estimated_probability = parsed["probability"]
        direction = parsed["direction"]
        # Edge is always positive when there is an exploitable advantage in the
        # signal's direction.  For YES signals: edge = prob - mid (we think YES
        # is underpriced).  For NO signals: edge = (1 - prob) - (1 - mid) =
        # mid - prob (we think NO is underpriced).  SKIP signals keep the raw
        # YES-side value for audit purposes.
        if direction == "NO":
            edge = market.mid_price - estimated_probability
        else:
            edge = estimated_probability - market.mid_price
        docs = [doc for doc, _ in doc_pairs]

        signal_row = SignalRow(
            id=signal_id,
            market_id=market.id,
            estimated_probability=estimated_probability,
            confidence=parsed["confidence"],
            edge=edge,
            market_mid_at_signal=market.mid_price,
            direction=direction,
            reasoning=parsed["reasoning"],
            sources=[d.id for d in docs],
            retrieval_hash=retrieval_hash,
            model_used=self._model,
            prompt_version=PROMPT_VERSION,
            trigger=trigger,
            raw_context=raw_context,
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

        now = datetime.now(timezone.utc)
        new_id = uuid.uuid4()

        if current.direction == "NO":
            new_edge = market.mid_price - current.estimated_probability
        else:
            new_edge = current.estimated_probability - market.mid_price

        signal_row = SignalRow(
            id=new_id,
            market_id=market.id,
            estimated_probability=current.estimated_probability,
            confidence=current.confidence,
            edge=new_edge,
            market_mid_at_signal=market.mid_price,
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
