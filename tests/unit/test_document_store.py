"""Unit tests for freqpred/ingestion/store.py.

All database and embedder calls are mocked — no external dependencies.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.store import RawDocument, UpsertStatus, _sha256, _strip_html, upsert_document
from freqpred.rag.models import DocumentRow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)

FAKE_EMBEDDING = [0.1] * 384


def _make_raw_doc(
    url: str = "https://example.com/article",
    body: str = "Plain body text.",
    **kwargs,
) -> RawDocument:
    defaults = dict(
        source_url=url,
        title="Example Article",
        body=body,
        source_type="news",
        source_name="Reuters",
        category="politics",
        tags=["election"],
        published_at=NOW,
        fetched_at=NOW,
    )
    defaults.update(kwargs)
    return RawDocument(**defaults)


def _make_document_row(source_url: str, content_hash: str) -> DocumentRow:
    import uuid

    return DocumentRow(
        id=uuid.uuid4(),
        source_url=source_url,
        content_hash=content_hash,
        title="Old Title",
        body="Old body",
        source_type="news",
        source_name="Reuters",
        category="politics",
        tags=[],
        published_at=NOW,
        fetched_at=NOW,
        embedding=FAKE_EMBEDDING,
        embedding_model="all-MiniLM-L6-v2",
    )


def _make_session(existing_row=None, upserted_row=None) -> AsyncMock:
    """Build a mock AsyncSession that returns *existing_row* on SELECT
    and *upserted_row* on the INSERT … ON CONFLICT upsert."""
    session = AsyncMock()

    # SELECT result (scalar_one_or_none)
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = existing_row

    # INSERT … RETURNING result (scalar_one)
    insert_result = MagicMock()
    insert_result.scalar_one.return_value = upserted_row

    # session.execute: first call → select_result, second → insert_result
    session.execute = AsyncMock(side_effect=[select_result, insert_result])
    session.flush = AsyncMock()

    return session


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------


def test_strip_html_removes_tags():
    assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_plain_text_unchanged():
    text = "No tags here."
    assert _strip_html(text) == text


def test_strip_html_multiline():
    html = "<div><p>Line one</p><p>Line two</p></div>"
    result = _strip_html(html)
    assert "Line one" in result
    assert "Line two" in result


# ---------------------------------------------------------------------------
# _sha256
# ---------------------------------------------------------------------------


def test_sha256_deterministic():
    text = "consistent input"
    assert _sha256(text) == _sha256(text)


def test_sha256_length():
    assert len(_sha256("anything")) == 64


def test_sha256_matches_hashlib():
    text = "test content"
    expected = hashlib.sha256(text.encode()).hexdigest()
    assert _sha256(text) == expected


# ---------------------------------------------------------------------------
# upsert_document — new document (no existing row)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_new_document_calls_embedder():
    """When no existing row, embedder must be called exactly once."""
    raw_doc = _make_raw_doc()
    body_clean = _strip_html(raw_doc.body)
    content_hash = _sha256(body_clean)

    upserted_row = _make_document_row(raw_doc.source_url, content_hash)
    session = _make_session(existing_row=None, upserted_row=upserted_row)

    embedder = AsyncMock()
    embedder.embed_text = AsyncMock(return_value=FAKE_EMBEDDING)

    doc, status = await upsert_document(session, embedder, raw_doc)

    embedder.embed_text.assert_awaited_once_with(body_clean)
    assert doc.source_url == raw_doc.source_url
    assert doc.content_hash == content_hash


@pytest.mark.asyncio
async def test_upsert_new_document_flushes_session():
    raw_doc = _make_raw_doc()
    body_clean = _strip_html(raw_doc.body)
    content_hash = _sha256(body_clean)

    upserted_row = _make_document_row(raw_doc.source_url, content_hash)
    session = _make_session(existing_row=None, upserted_row=upserted_row)
    embedder = AsyncMock()
    embedder.embed_text = AsyncMock(return_value=FAKE_EMBEDDING)

    await upsert_document(session, embedder, raw_doc)

    session.flush.assert_awaited_once()


# ---------------------------------------------------------------------------
# upsert_document — same URL, same content (no re-embed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_url_same_hash_skips_embed():
    """Inserting the same URL with identical content must not call embedder."""
    raw_doc = _make_raw_doc()
    body_clean = _strip_html(raw_doc.body)
    content_hash = _sha256(body_clean)

    existing_row = _make_document_row(raw_doc.source_url, content_hash)
    # Only one session.execute call expected (the SELECT); no INSERT.
    session = AsyncMock()
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = existing_row
    session.execute = AsyncMock(return_value=select_result)
    session.flush = AsyncMock()

    embedder = AsyncMock()
    embedder.embed_text = AsyncMock(return_value=FAKE_EMBEDDING)

    doc, status = await upsert_document(session, embedder, raw_doc)

    embedder.embed_text.assert_not_awaited()
    assert doc.source_url == raw_doc.source_url
    assert doc.content_hash == content_hash


# ---------------------------------------------------------------------------
# upsert_document — same URL, changed content (re-embed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_url_changed_hash_triggers_reembed():
    """Content change for an existing URL must call embedder once."""
    raw_doc = _make_raw_doc(body="Updated body text.")
    body_clean = _strip_html(raw_doc.body)
    new_hash = _sha256(body_clean)
    old_hash = _sha256("old content")

    existing_row = _make_document_row(raw_doc.source_url, old_hash)
    upserted_row = _make_document_row(raw_doc.source_url, new_hash)
    session = _make_session(existing_row=existing_row, upserted_row=upserted_row)

    embedder = AsyncMock()
    embedder.embed_text = AsyncMock(return_value=FAKE_EMBEDDING)

    doc, status = await upsert_document(session, embedder, raw_doc)

    embedder.embed_text.assert_awaited_once_with(body_clean)
    assert doc.content_hash == new_hash


# ---------------------------------------------------------------------------
# upsert_document — HTML stripping applied before hash / embed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_html_stripped_before_hashing():
    """HTML tags must be stripped from body before computing content_hash."""
    raw_doc_html = _make_raw_doc(body="<p>Hello <b>world</b></p>")
    raw_doc_plain = _make_raw_doc(body="Hello world")

    expected_hash = _sha256("Hello world")

    body_clean_html = _strip_html(raw_doc_html.body)
    body_clean_plain = _strip_html(raw_doc_plain.body)

    assert _sha256(body_clean_html) == expected_hash
    assert _sha256(body_clean_plain) == expected_hash


@pytest.mark.asyncio
async def test_embed_called_with_stripped_body():
    """embed_text must receive the plain-text body, not raw HTML."""
    raw_doc = _make_raw_doc(body="<p>Article <b>content</b></p>")
    plain_body = "Article content"
    content_hash = _sha256(plain_body)

    upserted_row = _make_document_row(raw_doc.source_url, content_hash)
    session = _make_session(existing_row=None, upserted_row=upserted_row)

    embedder = AsyncMock()
    embedder.embed_text = AsyncMock(return_value=FAKE_EMBEDDING)

    await upsert_document(session, embedder, raw_doc)

    embedder.embed_text.assert_awaited_once_with(plain_body)


# ---------------------------------------------------------------------------
# LocalEmbedder unit tests (mock the SentenceTransformer model)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_embedder_embed_text():
    """embed_text must return the first embedding from a single-text batch."""
    import numpy as np
    from freqpred.rag.embedder import LocalEmbedder

    with patch("freqpred.rag.embedder.SentenceTransformer") as MockST:
        mock_model = MockST.return_value
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([FAKE_EMBEDDING])

        embedder = LocalEmbedder()
        result = await embedder.embed_text("hello")

    assert result == FAKE_EMBEDDING
    mock_model.encode.assert_called_once()
    assert mock_model.encode.call_args.args[0] == ["hello"]


@pytest.mark.asyncio
async def test_local_embedder_embed_batch_empty():
    """embed_batch([]) must return [] without calling encode."""
    from freqpred.rag.embedder import LocalEmbedder

    with patch("freqpred.rag.embedder.SentenceTransformer") as MockST:
        mock_model = MockST.return_value
        mock_model.get_sentence_embedding_dimension.return_value = 384

        embedder = LocalEmbedder()
        result = await embedder.embed_batch([])

    assert result == []
    mock_model.encode.assert_not_called()


# ---------------------------------------------------------------------------
# Summarization behaviour in upsert_document
# ---------------------------------------------------------------------------

_LONG_BODY = "Federal Reserve holds interest rates steady. " * 30  # ~1350 chars
_MARKET_Q = "Will the Federal Reserve raise rates before June 2026?\nIf yes this market resolves Yes."


def _make_session_with_bm25(existing_row=None, bm25_score: float = 0.5, upserted_row=None) -> AsyncMock:
    """Session mock that handles the dedup SELECT, optional BM25 query, and INSERT."""
    session = AsyncMock()
    session.begin_nested = MagicMock()

    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = existing_row

    bm25_result = MagicMock()
    bm25_result.scalar.return_value = bm25_score

    insert_result = MagicMock()
    insert_result.scalar_one.return_value = upserted_row

    session.execute = AsyncMock(side_effect=[select_result, bm25_result, insert_result])
    session.flush = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_upsert_uses_summary_for_embedding_when_present():
    """When raw_doc.summary is pre-set, embed_text must use the summary."""
    raw_doc = _make_raw_doc(summary="Pre-existing summary text.")
    body_clean = _strip_html(raw_doc.body)
    content_hash = _sha256(body_clean)

    upserted_row = _make_document_row(raw_doc.source_url, content_hash)
    session = _make_session(existing_row=None, upserted_row=upserted_row)

    embedder = AsyncMock()
    embedder.embed_text = AsyncMock(return_value=FAKE_EMBEDDING)

    await upsert_document(session, embedder, raw_doc)

    embedder.embed_text.assert_awaited_once_with("Pre-existing summary text.")


@pytest.mark.asyncio
async def test_upsert_long_body_with_llm_client_calls_summarizer():
    """Long body + llm_client + market_question above BM25 threshold → summarize_body called."""
    raw_doc = _make_raw_doc(body=_LONG_BODY)
    body_clean = _strip_html(raw_doc.body)
    content_hash = _sha256(body_clean)

    upserted_row = _make_document_row(raw_doc.source_url, content_hash)
    session = _make_session_with_bm25(existing_row=None, bm25_score=0.3, upserted_row=upserted_row)

    embedder = AsyncMock()
    embedder.embed_text = AsyncMock(return_value=FAKE_EMBEDDING)

    llm_client = MagicMock()

    with patch(
        "freqpred.ingestion.body_summarizer.summarize_body",
        new=AsyncMock(return_value="Fed held rates steady in March 2026."),
    ):
        doc, status = await upsert_document(
            session, embedder, raw_doc,
            llm_client=llm_client,
            query_text="Fed interest rate",
            market_question=_MARKET_Q,
        )

    embedder.embed_text.assert_awaited_once_with("Fed held rates steady in March 2026.")
    assert doc.source_url == raw_doc.source_url


@pytest.mark.asyncio
async def test_upsert_deduped_doc_never_calls_summarizer():
    """Deduped document (same content_hash) must never trigger LLM summarization."""
    raw_doc = _make_raw_doc(body=_LONG_BODY)
    body_clean = _strip_html(raw_doc.body)
    content_hash = _sha256(body_clean)

    existing_row = _make_document_row(raw_doc.source_url, content_hash)
    session = AsyncMock()
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = existing_row
    session.execute = AsyncMock(return_value=select_result)
    session.flush = AsyncMock()

    embedder = AsyncMock()
    embedder.embed_text = AsyncMock(return_value=FAKE_EMBEDDING)
    llm_client = MagicMock()

    with patch(
        "freqpred.ingestion.body_summarizer.summarize_body",
        new=AsyncMock(return_value="should not be called"),
    ) as mock_summarize:
        _, status = await upsert_document(
            session, embedder, raw_doc,
            llm_client=llm_client,
            query_text="Fed interest rate",
            market_question=_MARKET_Q,
        )

    assert status == UpsertStatus.DEDUPED
    mock_summarize.assert_not_awaited()
    embedder.embed_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_upsert_low_bm25_skips_summarizer():
    """Body above threshold but BM25 score below minimum → summarize_body not called."""
    raw_doc = _make_raw_doc(body=_LONG_BODY)
    body_clean = _strip_html(raw_doc.body)
    content_hash = _sha256(body_clean)

    upserted_row = _make_document_row(raw_doc.source_url, content_hash)
    session = _make_session_with_bm25(existing_row=None, bm25_score=0.001, upserted_row=upserted_row)

    embedder = AsyncMock()
    embedder.embed_text = AsyncMock(return_value=FAKE_EMBEDDING)
    llm_client = MagicMock()

    with patch(
        "freqpred.ingestion.body_summarizer.summarize_body",
        new=AsyncMock(return_value="should not be called"),
    ) as mock_summarize:
        await upsert_document(
            session, embedder, raw_doc,
            llm_client=llm_client,
            query_text="unrelated topic",
            market_question=_MARKET_Q,
        )

    mock_summarize.assert_not_awaited()
    # Embedding should still happen, using body (no summary)
    embedder.embed_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_embedder_embed_batch_multiple():
    """embed_batch must return one vector per input text."""
    import numpy as np
    from freqpred.rag.embedder import LocalEmbedder

    texts = ["text 1", "text 2", "text 3"]
    fake_matrix = np.array([[0.1] * 384, [0.2] * 384, [0.3] * 384])

    with patch("freqpred.rag.embedder.SentenceTransformer") as MockST:
        mock_model = MockST.return_value
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = fake_matrix

        embedder = LocalEmbedder()
        result = await embedder.embed_batch(texts)

    assert len(result) == 3
    assert result[0] == [0.1] * 384
