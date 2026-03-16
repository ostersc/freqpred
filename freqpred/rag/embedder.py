"""Voyage AI embedding client for voyage-3 (1024-dim)."""
from __future__ import annotations

import structlog
import voyageai

log = structlog.get_logger()

_VOYAGE_MODEL = "voyage-3"
_VOYAGE_DIM = 1024
_VOYAGE_BATCH_SIZE = 128  # Voyage AI max texts per request


class VoyageEmbedder:
    """Async Voyage AI client wrapping voyage-3 embeddings (dim=1024)."""

    def __init__(self, api_key: str) -> None:
        self._client = voyageai.AsyncClient(api_key=api_key)
        self.model = _VOYAGE_MODEL
        self.dim = _VOYAGE_DIM

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string. Returns a 1024-dim float list."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching to respect API limits.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text, each of length 1024.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _VOYAGE_BATCH_SIZE):
            chunk = texts[i : i + _VOYAGE_BATCH_SIZE]
            log.debug("voyage.embed_batch", chunk_size=len(chunk), model=self.model)
            result = await self._client.embed(chunk, model=self.model)
            all_embeddings.extend(result.embeddings)

        return all_embeddings
