"""Tests for semantic similarity utilities."""

from __future__ import annotations

import numpy as np

from nlp import similarity
from parser.schemas import Resume


def test_calculate_resume_job_similarity_returns_bounded_score(monkeypatch) -> None:
    """Compute cosine similarity between resume and job-description embeddings."""

    def fake_get_embedding(text: str) -> np.ndarray:
        if "Python" in text:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.5, 0.5], dtype=np.float32)

    monkeypatch.setattr(similarity, "get_embedding", fake_get_embedding)
    resume = Resume(
        name="Aisha Khan",
        email="aisha@example.com",
        phone="555-555-5555",
        skills=["Python"],
    )

    score = similarity.calculate_resume_job_similarity(
        resume,
        "Python developer with SQL experience",
    )

    assert 0.0 <= score <= 1.0
    assert score == 1.0


def test_calculate_cosine_similarity_handles_zero_vectors() -> None:
    """Return zero for zero-vector comparisons."""
    score = similarity.calculate_cosine_similarity(
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
    )

    assert score == 0.0

