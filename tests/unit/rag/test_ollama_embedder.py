"""Unit tests for OllamaEmbedder in freqpred/rag/embedder.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.rag.embedder import OllamaEmbedder

# ---------------------------------------------------------------------------
# test_embed_text_calls_ollama
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_text_calls_ollama() -> None:
    """OllamaEmbedder POSTs correct JSON to /api/embed and returns the embedding."""
    fake_vector = [0.1, 0.2, 0.3]
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [fake_vector]}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    embedder = OllamaEmbedder(model="nomic-embed-text", base_url="http://localhost:11434")

    with patch("freqpred.rag.embedder.httpx.AsyncClient", return_value=mock_client):
        result = await embedder.embed_text("test query")

    assert result == fake_vector

    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    url = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("url", "")
    assert "/api/embed" in url

    payload = call_kwargs.kwargs.get("json") or (call_kwargs.args[1] if len(call_kwargs.args) > 1 else {})
    assert payload["model"] == "nomic-embed-text"
    assert payload["input"] == "test query"
    assert payload["truncate"] is True


# ---------------------------------------------------------------------------
# test_embed_text_raises_on_500
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_text_raises_on_500() -> None:
    """OllamaEmbedder raises RuntimeError with status and detail on non-200 response."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "model not found"

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    embedder = OllamaEmbedder()

    with patch("freqpred.rag.embedder.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(RuntimeError, match="500"):
            await embedder.embed_text("test")


# ---------------------------------------------------------------------------
# test_embed_text_protocol
# ---------------------------------------------------------------------------


def test_embed_text_protocol() -> None:
    """OllamaEmbedder structurally satisfies the Embedder protocol."""
    embedder = OllamaEmbedder()
    assert hasattr(embedder, "embed_text")
    assert callable(embedder.embed_text)
    assert isinstance(embedder.model_name, str)
    assert isinstance(embedder.max_embed_chars, int)
    assert embedder.embedding_column == "embedding_768"
