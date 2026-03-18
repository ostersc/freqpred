"""Local sentence-transformers embedder (free, no API key required).

Uses ``all-MiniLM-L6-v2`` by default — 384-dim, ~90 MB, runs on CPU.
The model is downloaded from HuggingFace on first use and cached locally.

Public API:
    LocalEmbedder  — async wrapper around SentenceTransformer
"""
from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

log = structlog.get_logger()

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Single shared thread pool for CPU-bound inference — avoids spinning up a
# new thread per call while still keeping the event loop unblocked.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedder")


class LocalEmbedder:
    """Async wrapper around SentenceTransformer for drop-in embedding support.

    All encoding runs in a thread-pool executor so the asyncio event loop is
    never blocked, even on CPU-only hardware.

    Args:
        model_name: HuggingFace model ID. Defaults to ``all-MiniLM-L6-v2``
                    (384-dim, ~90 MB).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        log.info("embedder.loading_model", model=model_name)
        self._model = SentenceTransformer(model_name)
        self.model = model_name
        self.dim: int = self._model.get_sentence_embedding_dimension()
        log.info("embedder.model_loaded", model=model_name, dim=self.dim)

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single string. Returns a dim-length float list."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings. Returns one vector per input text."""
        if not texts:
            return []

        loop = asyncio.get_event_loop()
        encode_fn = functools.partial(
            self._model.encode,
            texts,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings: np.ndarray = await loop.run_in_executor(_EXECUTOR, encode_fn)
        return embeddings.tolist()
