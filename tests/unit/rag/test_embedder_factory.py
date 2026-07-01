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
