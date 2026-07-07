"""Tests for Sentence Transformer embedding utilities."""

from __future__ import annotations

import numpy as np

from nlp import embeddings


class _DummySentenceTransformer:
    """Small test double for SentenceTransformer."""

    calls = 0

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        _DummySentenceTransformer.calls += 1

    def encode(
        self,
        text: str,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)


def test_get_embedding_returns_numpy_vector_and_loads_model_once(monkeypatch) -> None:
    """Generate embeddings through a singleton model instance."""
    _DummySentenceTransformer.calls = 0
    embeddings.get_embedding_model.cache_clear()
    monkeypatch.setattr(embeddings, "SentenceTransformer", _DummySentenceTransformer)

    first_embedding = embeddings.get_embedding("Python machine learning")
    second_embedding = embeddings.get_embedding("SQL data engineering")

    assert isinstance(first_embedding, np.ndarray)
    assert first_embedding.shape == (3,)
    assert np.array_equal(first_embedding, second_embedding)
    assert _DummySentenceTransformer.calls == 1

    embeddings.get_embedding_model.cache_clear()

