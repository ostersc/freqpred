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

    The model is loaded lazily on the first encode call, not at construction
    time. This avoids triggering torch's multiprocessing initialisation at
    import time (which on macOS uses the ``spawn`` start method and can
    conflict with same-named packages in the project).

    Args:
        model_name: HuggingFace model ID. Defaults to ``all-MiniLM-L6-v2``
                    (384-dim, ~90 MB).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model = model_name
        self.dim: int = 0  # set after first load
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        """Load the model synchronously (called once from the thread pool)."""
        log.info("embedder.loading_model", model=self.model)
        m = SentenceTransformer(self.model)
        self.dim = m.get_sentence_embedding_dimension()
        log.info("embedder.model_loaded", model=self.model, dim=self.dim)
        return m

    async def _ensure_loaded(self) -> SentenceTransformer:
        if self._model is None:
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(_EXECUTOR, self._load)
        return self._model

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single string. Returns a dim-length float list."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings. Returns one vector per input text."""
        if not texts:
            return []

        model = await self._ensure_loaded()
        loop = asyncio.get_event_loop()
        encode_fn = functools.partial(
            model.encode,
            texts,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings: np.ndarray = await loop.run_in_executor(_EXECUTOR, encode_fn)
        return embeddings.tolist()
