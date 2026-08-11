"""Embedder implementations and factory.

Two backends:
    LocalEmbedder   — sentence-transformers, runs on CPU, no API key required.
    OllamaEmbedder  — delegates to a local Ollama server (e.g. nomic-embed-text).

Both satisfy the Embedder protocol defined in retriever.py:
    async def embed_text(self, text: str) -> list[float]

Both also expose:
    model_name: str        — written to documents.embedding_model
    max_embed_chars: int   — how many chars to truncate before embedding

Use make_embedder(config) to construct the right one from EmbeddingConfig.
"""
from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import httpx
import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from freqpred.config import EmbeddingConfig

log = structlog.get_logger()

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Per-backend truncation defaults, used when EmbeddingConfig.max_embed_chars is
# left unset. all-MiniLM-L6-v2 has a 512-token limit (~2K chars); nomic-embed-text
# has an 8K-token limit, and 8K chars is ~2K tokens — well inside it.
_DEFAULT_MAX_EMBED_CHARS = 2_000
_OLLAMA_MAX_EMBED_CHARS = 8_000

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
        model_name:      HuggingFace model ID. Defaults to ``all-MiniLM-L6-v2``
                         (384-dim, ~90 MB).
        max_embed_chars: Truncation limit before embedding. Defaults to 2000.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        max_embed_chars: int = _DEFAULT_MAX_EMBED_CHARS,
    ) -> None:
        self.model_name = model_name
        self.max_embed_chars = max_embed_chars
        self.embedding_column = "embedding"
        self.dim: int = 0  # set after first load
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        """Load the model synchronously (called once from the thread pool)."""
        log.info("embedder.loading_model", model=self.model_name)
        m = SentenceTransformer(self.model_name)
        self.dim = m.get_sentence_embedding_dimension()
        log.info("embedder.model_loaded", model=self.model_name, dim=self.dim)
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


class OllamaEmbedder:
    """Embedding via a local Ollama server (e.g. nomic-embed-text, 768-dim).

    POSTs to /api/embeddings with ``num_ctx`` set to 8192 so Ollama uses the
    full context window of nomic-embed-text (8K tokens vs the default 2K).

    Args:
        model:           Ollama model name. Defaults to ``nomic-embed-text``.
        base_url:        Ollama server base URL. Defaults to localhost:11434.
        max_embed_chars: Truncation limit before embedding. Defaults to 8000
                         (~2000 tokens, still far inside nomic's 8K window).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        max_embed_chars: int = _OLLAMA_MAX_EMBED_CHARS,
    ) -> None:
        self.model_name = model
        self.max_embed_chars = max_embed_chars
        self.embedding_column = "embedding_768"
        self._base_url = base_url.rstrip("/")

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single string via Ollama. Returns a float list.

        Uses the /api/embed endpoint (Ollama ≥ 0.1.26) with truncate=True so
        the model silently clips inputs that exceed its context window instead
        of returning a 500 error.
        """
        url = f"{self._base_url}/api/embed"
        payload = {
            "model": self.model_name,
            "input": text,
            "truncate": True,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            detail = resp.text[:200]
            raise RuntimeError(
                f"Ollama embedding failed ({resp.status_code}): {detail}"
            )
        return resp.json()["embeddings"][0]


def make_embedder(config: EmbeddingConfig) -> LocalEmbedder | OllamaEmbedder:
    """Construct the configured embedder from EmbeddingConfig.

    ``max_embed_chars`` is resolved per backend when the config leaves it unset,
    so switching to ollama does not silently keep MiniLM's 2K-char truncation.
    """
    if config.backend == "ollama":
        return OllamaEmbedder(
            model=config.model,
            base_url=config.ollama_base_url,
            max_embed_chars=config.max_embed_chars or _OLLAMA_MAX_EMBED_CHARS,
        )
    return LocalEmbedder(
        model_name=config.model,
        max_embed_chars=config.max_embed_chars or _DEFAULT_MAX_EMBED_CHARS,
    )
