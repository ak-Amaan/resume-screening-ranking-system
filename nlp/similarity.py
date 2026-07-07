"""Cosine similarity scoring utilities for resumes and job descriptions."""

from __future__ import annotations

import logging
from dataclasses import is_dataclass

import numpy as np

from nlp.embeddings import get_embedding
from parser.schemas import Resume

logger = logging.getLogger(__name__)


def calculate_cosine_similarity(first_vector: np.ndarray, second_vector: np.ndarray) -> float:
    """Calculate cosine similarity for two vectors as a 0.0 to 1.0 score.

    Args:
        first_vector: First embedding vector.
        second_vector: Second embedding vector.

    Returns:
        Cosine similarity clipped to the inclusive range [0.0, 1.0].
    """
    first = np.asarray(first_vector, dtype=np.float32).reshape(-1)
    second = np.asarray(second_vector, dtype=np.float32).reshape(-1)

    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0

    similarity = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.clip(similarity, 0.0, 1.0))


def calculate_resume_job_similarity(resume: Resume, job_description: str | object) -> float:
    """Calculate semantic similarity between a parsed resume and a job description.

    Args:
        resume: Parsed resume dataclass.
        job_description: Raw job description text or parsed job description object.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    resume_text = resume_to_text(resume)
    job_text = job_description_to_text(job_description)

    logger.info("Computing resume-job semantic similarity for %s", resume.name)
    resume_embedding = get_embedding(resume_text)
    job_embedding = get_embedding(job_text)
    return calculate_cosine_similarity(resume_embedding, job_embedding)


def resume_to_text(resume: Resume) -> str:
    """Convert a Resume dataclass into text for semantic comparison."""
    if resume.raw_text:
        return resume.raw_text

    education_text = " ".join(
        f"{item.degree} {item.institution}" for item in resume.education
    )
    certifications_text = " ".join(
        f"{item.name} {item.issuer}" for item in resume.certifications
    )
    projects_text = " ".join(
        f"{item.name} {item.description} {' '.join(item.technologies)}"
        for item in resume.projects
    )
    return " ".join(
        [
            resume.name,
            " ".join(resume.skills),
            education_text,
            " ".join(resume.experience),
            certifications_text,
            projects_text,
            str(resume.years_of_experience),
        ]
    )


def job_description_to_text(job_description: str | object) -> str:
    """Convert a raw or parsed job description into text."""
    if isinstance(job_description, str):
        return job_description

    if is_dataclass(job_description):
        values = []
        for field_name in getattr(job_description, "__dataclass_fields__", {}):
            value = getattr(job_description, field_name)
            if isinstance(value, list):
                values.append(" ".join(str(item) for item in value))
            else:
                values.append(str(value))
        return " ".join(values)

    return str(job_description)
