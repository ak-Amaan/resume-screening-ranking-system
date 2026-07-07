"""Sentence Transformer embedding utilities."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


class EmbeddingModelError(Exception):
    """Raised when the sentence transformer model cannot generate embeddings."""


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the Sentence Transformer model once and reuse it.

    Returns:
        Singleton instance of the configured Sentence Transformer model.

    Raises:
        EmbeddingModelError: If the model cannot be loaded.
    """
    try:
        logger.info("Loading Sentence Transformer model: %s", MODEL_NAME)
        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        logger.exception("Failed to load Sentence Transformer model: %s", MODEL_NAME)
        raise EmbeddingModelError(f"Unable to load embedding model: {MODEL_NAME}") from exc


def get_embedding(text: str) -> np.ndarray:
    """Generate a sentence embedding for text.

    Args:
        text: Input text to embed.

    Returns:
        One-dimensional numpy embedding vector.

    Raises:
        EmbeddingModelError: If embedding generation fails.
    """
    normalized_text = " ".join(text.split())
    if not normalized_text:
        normalized_text = " "

    try:
        model = get_embedding_model()
        embedding = model.encode(
            normalized_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except EmbeddingModelError:
        raise
    except Exception as exc:
        logger.exception("Failed to generate embedding.")
        raise EmbeddingModelError("Unable to generate embedding.") from exc

    return np.asarray(embedding, dtype=np.float32).reshape(-1)
