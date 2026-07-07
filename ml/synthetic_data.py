"""Synthetic resume and job-description data generation module."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

TARGET_COLUMN = "target_candidate_score"
DEFAULT_SYNTHETIC_DATA_PATH = Path("data/generated/synthetic_training_data.csv")
RANDOM_STATE = 42


def calculate_target_score(features: pd.DataFrame) -> pd.Series:
    """Calculate transparent synthetic candidate labels on a 0-100 scale.

    Formula:
        Candidate Score =
            0.40 * Semantic Similarity
          + 0.20 * Skill Match
          + 0.15 * Experience Match
          + 0.10 * Education Match
          + 0.05 * Certification Match
          + 0.05 * Programming Language Match
          + 0.03 * Framework Match
          + 0.02 * Tools Match

    The weighted sum is clipped to [0, 1] and multiplied by 100. This makes
    the target interpretable while still allowing a supervised model to learn
    non-linear patterns from the engineered feature vector.

    Args:
        features: DataFrame containing the Step 3 feature columns.

    Returns:
        Target candidate score between 0 and 100.
    """
    weighted_score = (
        0.40 * features["semantic_similarity"]
        + 0.20 * features["skill_match_percentage"]
        + 0.15 * features["experience_match_score"]
        + 0.10 * features["education_match_score"]
        + 0.05 * features["certification_match_score"]
        + 0.05 * features["programming_language_match"]
        + 0.03 * features["framework_match"]
        + 0.02 * features["tools_match"]
    )
    return weighted_score.clip(0.0, 1.0).mul(100.0)


def generate_synthetic_dataset(
    n_samples: int = 1_000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate realistic synthetic training data for candidate ranking.

    Args:
        n_samples: Number of candidate-job samples to generate.
        random_state: Seed for reproducible synthetic data.

    Returns:
        DataFrame with engineered resume/job features and target scores.

    Raises:
        ValueError: If fewer than one sample is requested.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    rng = np.random.default_rng(random_state)
    candidate_skill_counts = rng.integers(3, 16, size=n_samples)
    required_skill_counts = rng.integers(3, 12, size=n_samples)
    number_of_matching_skills = np.minimum(
        rng.binomial(required_skill_counts, rng.beta(2.5, 2.0, size=n_samples)),
        candidate_skill_counts,
    )
    skill_match = number_of_matching_skills / required_skill_counts

    semantic_similarity = np.clip(
        0.30 + 0.50 * skill_match + rng.normal(0.0, 0.12, size=n_samples),
        0.0,
        1.0,
    )
    required_years = rng.choice(
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=float),
        size=n_samples,
        p=[0.04, 0.08, 0.14, 0.18, 0.18, 0.14, 0.11, 0.08, 0.05],
    )
    candidate_years = np.clip(
        required_years + rng.normal(0.8, 2.0, size=n_samples),
        0.0,
        15.0,
    )
    experience_ratio = np.divide(
        candidate_years,
        required_years,
        out=np.ones_like(candidate_years),
        where=required_years > 0,
    )
    experience_match = np.minimum(experience_ratio, 1.0)
    years_difference = np.abs(candidate_years - required_years)

    education_match = rng.binomial(
        1,
        np.clip(0.45 + 0.30 * semantic_similarity, 0.0, 1.0),
        size=n_samples,
    )
    certification_match = rng.binomial(
        1,
        np.clip(0.25 + 0.35 * skill_match, 0.0, 1.0),
        size=n_samples,
    )
    programming_language_match = np.clip(
        skill_match + rng.normal(0.05, 0.18, size=n_samples),
        0.0,
        1.0,
    )
    framework_match = np.clip(
        skill_match + rng.normal(-0.05, 0.22, size=n_samples),
        0.0,
        1.0,
    )
    tools_match = np.clip(
        skill_match + rng.normal(0.00, 0.20, size=n_samples),
        0.0,
        1.0,
    )
    project_relevance = np.clip(
        0.20 + 0.45 * semantic_similarity + 0.25 * skill_match
        + rng.normal(0.0, 0.12, size=n_samples),
        0.0,
        1.0,
    )

    features = pd.DataFrame(
        {
            "semantic_similarity": semantic_similarity,
            "skill_match_percentage": skill_match,
            "experience_match_score": experience_match,
            "education_match_score": education_match.astype(float),
            "certification_match_score": certification_match.astype(float),
            "project_relevance_score": project_relevance,
            "programming_language_match": programming_language_match,
            "framework_match": framework_match,
            "tools_match": tools_match,
            "years_of_experience_difference": years_difference,
            "number_of_matching_skills": number_of_matching_skills.astype(float),
            "total_candidate_skills": candidate_skill_counts.astype(float),
        },
        columns=FEATURE_COLUMNS,
    )
    target = calculate_target_score(features)
    target += rng.normal(0.0, 2.0, size=n_samples)

    dataset = features.copy()
    dataset[TARGET_COLUMN] = target.clip(0.0, 100.0)
    return dataset


def save_synthetic_dataset(
    dataset: pd.DataFrame,
    output_path: str | Path = DEFAULT_SYNTHETIC_DATA_PATH,
) -> Path:
    """Save a synthetic dataset as CSV.

    Args:
        dataset: Synthetic training dataset.
        output_path: Destination CSV path.

    Returns:
        Path to the saved dataset.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    logger.info("Saved synthetic training dataset to %s", path)
    return path


def load_or_generate_synthetic_dataset(
    data_path: str | Path = DEFAULT_SYNTHETIC_DATA_PATH,
    n_samples: int = 1_000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Load an existing synthetic dataset or create one if missing."""
    path = Path(data_path)
    if path.exists():
        logger.info("Loading synthetic training dataset from %s", path)
        return pd.read_csv(path)

    logger.info("Generating %d synthetic candidate samples.", n_samples)
    dataset = generate_synthetic_dataset(
        n_samples=n_samples,
        random_state=random_state,
    )
    save_synthetic_dataset(dataset, path)
    return dataset
