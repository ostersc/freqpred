"""Unit tests for the make_embedder factory in freqpred/rag/embedder.py."""
from __future__ import annotations

from freqpred.config import EmbeddingConfig
from freqpred.rag.embedder import LocalEmbedder, OllamaEmbedder, make_embedder


def test_factory_returns_local_embedder_for_sentence_transformers() -> None:
    cfg = EmbeddingConfig(
        backend="sentence_transformers",
        model="all-MiniLM-L6-v2",
        max_embed_chars=2000,
    )
    embedder = make_embedder(cfg)
    assert isinstance(embedder, LocalEmbedder)
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.max_embed_chars == 2000
    assert embedder.embedding_column == "embedding"


def test_factory_returns_ollama_embedder_for_ollama() -> None:
    cfg = EmbeddingConfig(
        backend="ollama",
        model="nomic-embed-text",
        ollama_base_url="http://localhost:11434",
        max_embed_chars=6000,
    )
    embedder = make_embedder(cfg)
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model_name == "nomic-embed-text"
    assert embedder.max_embed_chars == 6000
    assert embedder.embedding_column == "embedding_768"


def test_local_embedder_exposes_protocol_attributes() -> None:
    embedder = LocalEmbedder(model_name="test-model", max_embed_chars=1234)
    assert embedder.model_name == "test-model"
    assert embedder.max_embed_chars == 1234
    assert embedder.embedding_column == "embedding"
    assert callable(embedder.embed_text)


def test_ollama_embedder_exposes_protocol_attributes() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text", max_embed_chars=5000)
    assert embedder.model_name == "nomic-embed-text"
    assert embedder.max_embed_chars == 5000
    assert embedder.embedding_column == "embedding_768"
    assert callable(embedder.embed_text)


def test_factory_default_config_uses_sentence_transformers() -> None:
    cfg = EmbeddingConfig()
    embedder = make_embedder(cfg)
    assert isinstance(embedder, LocalEmbedder)
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.max_embed_chars == 2000
    assert embedder.embedding_column == "embedding"


def test_ollama_max_embed_chars_defaults_to_8000_when_config_unset() -> None:
    """Unset max_embed_chars must resolve per backend, not fall back to MiniLM's 2000.

    Before T100 the config default (2000) applied to every backend, so a config
    selecting ollama without naming max_embed_chars silently truncated at 2K chars
    despite nomic's 8K-token window.
    """
    cfg = EmbeddingConfig(backend="ollama", model="nomic-embed-text")
    embedder = make_embedder(cfg)
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.max_embed_chars == 8000


def test_ollama_embedder_default_max_embed_chars_is_8000() -> None:
    assert OllamaEmbedder().max_embed_chars == 8000


def test_explicit_max_embed_chars_overrides_backend_default() -> None:
    cfg = EmbeddingConfig(backend="ollama", model="nomic-embed-text", max_embed_chars=3000)
    assert make_embedder(cfg).max_embed_chars == 3000
